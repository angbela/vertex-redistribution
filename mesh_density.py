import ezdxf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import LineString, Point
import sys
import os

COLOR_MAP = {
    1: '#ff0000', 2: '#ffff00', 3: '#00ff00', 4: '#00ffff',
    5: '#0000ff', 6: '#ff00ff', 7: '#ffffff'
}

def dxf_color_to_hex(color_idx, fallback='#888888'):
    if color_idx is None or color_idx == 256:
        return fallback
    if color_idx == 0:
        return '#000000'
    if 1 <= color_idx <= 7:
        return COLOR_MAP[color_idx]
    return fallback

def bulge_points_from_lwpolyline(entity, arc_seg=20):
    pts_xy = list(entity.get_points('xy'))
    pts_xyse = list(entity.get_points('xyse'))
    closed = entity.closed
    coords = []
    n = len(pts_xy)
    for i in range(n):
        x1, y1 = pts_xy[i][0], pts_xy[i][1]
        bulge = 0.0
        if i < len(pts_xyse):
            bulge = pts_xyse[i][2] if len(pts_xyse[i]) > 2 else 0.0
        if not closed and i == n - 1:
            coords.append((x1, y1))
            continue
        if closed and i == n - 1 and not closed:
            coords.append((x1, y1))
            continue
        j = (i + 1) % n if closed else i + 1
        if j >= n:
            coords.append((x1, y1))
            continue
        x2, y2 = pts_xy[j][0], pts_xy[j][1]
        coords.append((x1, y1))
        if abs(bulge) > 1e-9:
            cx, cy, R, a1, a2, ccw = _bulge_to_arc(x1, y1, x2, y2, bulge)
            theta1 = np.arctan2(y1 - cy, x1 - cx)
            theta2 = np.arctan2(y2 - cy, x2 - cx)
            if ccw:
                if theta2 < theta1:
                    theta2 += 2 * np.pi
            else:
                if theta2 > theta1:
                    theta2 -= 2 * np.pi
            n_steps = max(2, int(abs(theta2 - theta1) * R / (0.5)))
            n_steps = max(n_steps, arc_seg // 2)
            dtheta = (theta2 - theta1) / n_steps
            for k in range(1, n_steps):
                t = theta1 + k * dtheta
                coords.append((cx + R * np.cos(t), cy + R * np.sin(t)))
    return coords

def _bulge_to_arc(x1, y1, x2, y2, B):
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx * dx + dy * dy)
    if d < 1e-14:
        return x1, y1, 0, 0, 0, True
    h = (d * B) / 2.0
    h_sq = h * h
    half_d = d / 2.0
    denom = half_d * half_d + h_sq
    if abs(h_sq) < 1e-15:
        return x1, y1, 0, 0, 0, True
    R = denom / (2.0 * abs(h))
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    nx = -dy / d
    ny = dx / d
    sign = 1.0 if B > 0 else -1.0
    dist_center = np.sqrt(max(0.0, R * R - (d / 2.0) ** 2))
    cx = mx + sign * dist_center * nx
    cy = my + sign * dist_center * ny
    a1 = np.arctan2(y1 - cy, x1 - cx)
    a2 = np.arctan2(y2 - cy, x2 - cx)
    return cx, cy, R, a1, a2, B > 0

def extract_polylines(filepath, target_layers=None):
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    polylines = []
    for e in msp:
        etype = e.dxftype()
        layer = e.dxf.layer
        if target_layers and layer not in target_layers:
            continue
        if etype not in ('LWPOLYLINE', 'POLYLINE', 'LINE'):
            continue
        coords = []
        if etype == 'LWPOLYLINE':
            coords = bulge_points_from_lwpolyline(e)
        elif etype == 'POLYLINE':
            pts = list(e.points())
            coords = [(p[0], p[1]) for p in pts]
            if e.is_closed and len(coords) > 0:
                coords.append(coords[0])
        elif etype == 'LINE':
            coords = [(e.dxf.start[0], e.dxf.start[1]), (e.dxf.end[0], e.dxf.end[1])]
        if len(coords) < 2:
            continue
        color = e.dxf.color
        rgb = dxf_color_to_hex(color)
        layer_color = None
        try:
            l = doc.layers.get(layer)
            lc = l.dxf.color
            layer_color = dxf_color_to_hex(lc)
        except Exception:
            pass
        plot_color = layer_color if layer_color else rgb
        polylines.append({
            'layer': layer,
            'color_idx': color,
            'color': plot_color,
            'coords': np.array(coords, dtype=np.float64),
            'entity': e,
        })
    return polylines

def classify_polyline_category(poly):
    layer = poly['layer'].lower()
    color = poly['color'].lower()
    if 'batas' in layer or 'pantai' in layer:
        return 'coastline'
    if color.startswith('#ff00') or color == '#ff0000':
        return 'boundary'
    cidx = poly.get('color_idx')
    if cidx == 1:
        return 'boundary'
    if cidx == 3:
        return 'coastline'
    return 'other'

def split_domain_polyline(poly):
    coords = poly['coords']
    n = len(coords)
    if n < 10:
        return [poly]
    closed = np.allclose(coords[0], coords[-1])
    if not closed:
        return [poly]
    diffs = np.diff(coords[:-1], axis=0)
    dirs = np.arctan2(diffs[:, 1], diffs[:, 0])
    ddir = np.abs(np.diff(dirs))
    ddir = np.concatenate([ddir, [np.abs(dirs[0] - dirs[-1])]])
    corner_idx = np.where(ddir > 0.3)[0]
    if len(corner_idx) < 2:
        return [poly]
    segments = []
    start = 0
    for ci in corner_idx:
        end = ci + 2
        seg_coords = coords[start:end]
        if len(seg_coords) >= 2:
            segments.append(seg_coords)
        start = ci + 1
    if start < n - 1:
        seg_coords = coords[start:n]
        if len(seg_coords) >= 2:
            segments.append(seg_coords)
    result = []
    for sc in segments:
        slen = np.sum(np.linalg.norm(np.diff(sc, axis=0), axis=1))
        seg_diffs = np.diff(sc, axis=0)
        seg_dirs = np.arctan2(seg_diffs[:, 1], seg_diffs[:, 0])
        dir_var = np.std(seg_dirs) if len(seg_dirs) > 1 else 0
        is_straight = dir_var < 0.05
        result.append({
            'layer': poly['layer'],
            'color_idx': poly['color_idx'],
            'color': '#ff0000' if is_straight else poly['color'],
            'coords': sc,
            'category': 'boundary' if is_straight else 'coastline',
            'entity': poly['entity'],
        })
    return result

def interactive_pick_points(ax, fig, polylines, focus_radius_guess=100):
    for poly in polylines:
        cat = classify_polyline_category(poly)
        color = poly['color']
        lw = 1.5 if cat == 'coastline' else 2.0
        if cat == 'boundary':
            color = '#ff0000'
        ax.plot(poly['coords'][:, 0], poly['coords'][:, 1], color=color, linewidth=lw, label=f"{poly['layer']}:{cat}")
    ax.set_aspect('equal')
    ax.set_title('Step 1: Click FOCUS CENTER (port zone) - then close window', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)
    print("\n>>> Click 1/4: Click the FOCUS CENTER point (center of port zone). Close window when done.")
    pts = plt.ginput(n=1, timeout=0)
    if not pts:
        raise ValueError("No point selected")
    focus_center = np.array([pts[0][0], pts[0][1]])
    circ = Circle(focus_center, focus_radius_guess, color='orange', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circ)
    ax.plot(focus_center[0], focus_center[1], 'o', color='orange', markersize=10, label='Focus Center')
    ax.set_title('Step 2: Click point on edge of FOCUS ZONE (defines min-spacing radius) - close window', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    print(">>> Click 2/4: Click point at the EDGE OF FOCUS ZONE (this sets the radius where spacing is minimum). Close window when done.")
    pts2 = plt.ginput(n=1, timeout=0)
    if not pts2:
        raise ValueError("No point selected")
    focus_edge_pt = np.array([pts2[0][0], pts2[0][1]])
    r_min = float(np.linalg.norm(focus_edge_pt - focus_center))
    circ.remove()
    circ2 = Circle(focus_center, r_min, color='orange', fill=True, alpha=0.15, ec='orange', linestyle='--', label=f'Focus Zone (r={r_min:.1f})')
    ax.add_patch(circ2)
    ax.plot([focus_center[0], focus_edge_pt[0]], [focus_center[1], focus_edge_pt[1]], 'o-', color='orange', linewidth=2)
    ax.set_title('Step 3: Click point on BOUNDARY (defines max-spacing distance) - close window', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    print(">>> Click 3/4: Click point on the RED BOUNDARY (this sets the transition outer distance). Close window when done.")
    pts3 = plt.ginput(n=1, timeout=0)
    if not pts3:
        raise ValueError("No point selected")
    boundary_pt = np.array([pts3[0][0], pts3[0][1]])
    r_max = float(np.linalg.norm(boundary_pt - focus_center))
    ax.plot([focus_center[0], boundary_pt[0]], [focus_center[1], boundary_pt[1]], 's-', color='red', linewidth=2)
    circ3 = Circle(focus_center, r_max, color='red', fill=False, linestyle=':', alpha=0.4, label=f'Outer Dist (r={r_max:.1f})')
    ax.add_patch(circ3)
    ax.set_title('Step 4: Close window to continue, then enter spacing values in console', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    print(">>> Close the plot window, then enter spacing values in the console.")
    plt.show()
    print(f"\n=== Captured parameters:")
    print(f"  Focus center: ({focus_center[0]:.2f}, {focus_center[1]:.2f})")
    print(f"  Focus radius (r_min distance): {r_min:.2f}")
    print(f"  Boundary distance (r_max distance): {r_max:.2f}")
    min_spacing = float(input(f"\n  Enter MIN spacing inside focus zone [e.g. 10]: ") or "10")
    max_spacing = float(input(f"  Enter MAX spacing at boundary [e.g. 200]: ") or "200")
    return {
        'focus_center': focus_center,
        'r_min_dist': r_min,
        'r_max_dist': r_max,
        'min_spacing': min_spacing,
        'max_spacing': max_spacing,
    }

def spacing_field(params, x, y):
    fc = params['focus_center']
    d = np.sqrt((x - fc[0]) ** 2 + (y - fc[1]) ** 2)
    r_min = params['r_min_dist']
    r_max = params['r_max_dist']
    s_min = params['min_spacing']
    s_max = params['max_spacing']
    if d <= r_min:
        return s_min
    if d >= r_max:
        return s_max
    t = (d - r_min) / (r_max - r_min)
    t_smooth = t * t * (3 - 2 * t)
    return s_min + (s_max - s_min) * t_smooth

def redistribute_polyline(coords, params):
    if len(coords) < 2:
        return coords
    diffs = np.diff(coords, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-9:
        return coords
    new_pts = [coords[0].copy()]
    s = 0.0
    s_target = spacing_field(params, coords[0][0], coords[0][1])
    max_iter = int(total_len / max(params['min_spacing'], 1e-3) * 10 + 10000)
    iter_count = 0
    while s < total_len and iter_count < max_iter:
        iter_count += 1
        seg_idx = np.searchsorted(cum_len, s) - 1
        seg_idx = max(0, min(seg_idx, len(coords) - 2))
        seg_start = cum_len[seg_idx]
        seg_len = seg_lens[seg_idx]
        if seg_len < 1e-12:
            s = cum_len[seg_idx + 1]
            continue
        t_seg = (s - seg_start) / seg_len
        px = coords[seg_idx][0] + t_seg * diffs[seg_idx][0]
        py = coords[seg_idx][1] + t_seg * diffs[seg_idx][1]
        local_spacing = spacing_field(params, px, py)
        next_s = s + local_spacing
        if next_s >= total_len:
            break
        while True:
            seg_idx2 = np.searchsorted(cum_len, next_s) - 1
            seg_idx2 = max(0, min(seg_idx2, len(coords) - 2))
            ss = cum_len[seg_idx2]
            sl = seg_lens[seg_idx2]
            if sl < 1e-12:
                break
            tt = (next_s - ss) / sl
            tt = max(0.0, min(1.0, tt))
            ppx = coords[seg_idx2][0] + tt * diffs[seg_idx2][0]
            ppy = coords[seg_idx2][1] + tt * diffs[seg_idx2][1]
            corrected_spacing = spacing_field(params, ppx, ppy)
            if abs(corrected_spacing - local_spacing) / local_spacing < 0.05:
                break
            local_spacing = corrected_spacing
            next_s = s + corrected_spacing
            if next_s >= total_len:
                break
        if next_s >= total_len:
            break
        s = next_s
        seg_idx3 = np.searchsorted(cum_len, s) - 1
        seg_idx3 = max(0, min(seg_idx3, len(coords) - 2))
        ss3 = cum_len[seg_idx3]
        sl3 = seg_lens[seg_idx3]
        if sl3 > 1e-12:
            tt3 = (s - ss3) / sl3
            tt3 = max(0.0, min(1.0, tt3))
            nx = coords[seg_idx3][0] + tt3 * diffs[seg_idx3][0]
            ny = coords[seg_idx3][1] + tt3 * diffs[seg_idx3][1]
            new_pts.append(np.array([nx, ny]))
    new_pts.append(coords[-1].copy())
    return np.array(new_pts)

def find_connection_points(polylines, tol=1e-3):
    endpoints = []
    for i, poly in enumerate(polylines):
        c = poly['coords']
        endpoints.append((i, 'start', c[0]))
        endpoints.append((i, 'end', c[-1]))
    connections = []
    n_ep = len(endpoints)
    for a in range(n_ep):
        for b in range(a + 1, n_ep):
            ia, ta, pa = endpoints[a]
            ib, tb, pb = endpoints[b]
            if ia == ib:
                continue
            d = np.linalg.norm(pa - pb)
            if d < tol + 1e-6:
                continue
            if d < 5.0:
                connections.append((ia, ta, ib, tb, d))
    return connections

def snap_and_pin_endpoints(polylines, tol=5.0):
    connections = find_connection_points(polylines, tol=tol)
    for ia, ta, ib, tb, d in connections:
        ca = polylines[ia]['coords']
        cb = polylines[ib]['coords']
        pa = ca[0] if ta == 'start' else ca[-1]
        pb = cb[0] if tb == 'start' else cb[-1]
        mid = (pa + pb) / 2.0
        if ta == 'start':
            ca[0] = mid
        else:
            ca[-1] = mid
        polylines[ia]['coords'] = ca
        if tb == 'start':
            cb[0] = mid
        else:
            cb[-1] = mid
        polylines[ib]['coords'] = cb
    return polylines

def export_xyz(polylines, filepath):
    lines_out = []
    scatter_pts = []
    header_lines = []
    pt_idx = 0
    for i, poly in enumerate(polylines):
        coords = poly['coords']
        cat = poly.get('category', classify_polyline_category(poly))
        layer = poly['layer']
        header_lines.append(f"# Polyline {i}: layer='{layer}' category='{cat}' n_pts={len(coords)}")
        for p in coords:
            scatter_pts.append((p[0], p[1], 0.0))
    with open(filepath, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')
        f.write(f"# Total scatter points: {len(scatter_pts)}\n")
        f.write("# X Y Z\n")
        for p in scatter_pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return scatter_pts

def export_dxf(polylines, filepath):
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    used_layers = {}
    for poly in polylines:
        lname = poly['layer']
        if lname not in used_layers:
            try:
                doc.layers.add(lname)
            except Exception:
                pass
            used_layers[lname] = True
        coords = poly['coords']
        cat = poly.get('category', classify_polyline_category(poly))
        color = 1 if cat == 'boundary' else 7
        if len(coords) < 2:
            continue
        pts = [(float(c[0]), float(c[1])) for c in coords]
        lw = msp.add_lwpolyline(pts, close=False, dxfattribs={'layer': lname, 'color': color})
    doc.saveas(filepath)

def main():
    dxf_file = "domain + kontur.dxf"
    if not os.path.exists(dxf_file):
        dxf_file = "coastline.dxf"
    if not os.path.exists(dxf_file):
        print("No DXF file found!")
        sys.exit(1)
    print(f"Loading DXF: {dxf_file}")
    target_layers = ['domain', 'Batas Pantai', 'batas pantai', 'Daratan', 'Kontur']
    polylines = extract_polylines(dxf_file, target_layers=target_layers)
    print(f"Loaded {len(polylines)} polylines")
    all_polys = []
    for p in polylines:
        if p['layer'].lower() == 'domain':
            segs = split_domain_polyline(p)
            all_polys.extend(segs)
        else:
            cat = classify_polyline_category(p)
            p['category'] = cat
            if cat == 'boundary':
                p['color'] = '#ff0000'
            all_polys.append(p)
    polylines = all_polys
    print(f"After domain split: {len(polylines)} segments")
    for i, p in enumerate(polylines):
        cat = p.get('category', '?')
        print(f"  {i:2d}: layer={p['layer']:15s} cat={cat:12s} n={len(p['coords']):4d} color={p['color']}")
    fig, ax = plt.subplots(figsize=(12, 10))
    params = interactive_pick_points(ax, fig, polylines)
    print(f"\n=== Running redistribution...")
    all_x = np.concatenate([p['coords'][:, 0] for p in polylines])
    all_y = np.concatenate([p['coords'][:, 1] for p in polylines])
    diag_guess = float(np.sqrt((all_x.max() - all_x.min()) ** 2 + (all_y.max() - all_y.min()) ** 2) / 50.0)
    redistributed = []
    total_pts_before = 0
    total_pts_after = 0
    for p in polylines:
        coords = p['coords']
        total_pts_before += len(coords)
        new_coords = redistribute_polyline(coords, params)
        total_pts_after += len(new_coords)
        p_new = dict(p)
        p_new['coords'] = new_coords
        redistributed.append(p_new)
    print(f"  Points before: {total_pts_before}")
    print(f"  Points after:  {total_pts_after}")
    redistributed = snap_and_pin_endpoints(redistributed)
    xyz_path = "redistributed_points.xyz"
    export_xyz(redistributed, xyz_path)
    print(f"\n  Exported XYZ: {xyz_path}")
    dxf_out = "redistributed.dxf"
    export_dxf(redistributed, dxf_out)
    print(f"  Exported DXF: {dxf_out}")
    fig2, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax1, ax2 = axes
    for p in polylines:
        cat = p.get('category', classify_polyline_category(p))
        c = '#ff0000' if cat == 'boundary' else '#000000'
        ax1.plot(p['coords'][:, 0], p['coords'][:, 1], color=c, linewidth=1.0)
        ax1.plot(p['coords'][:, 0], p['coords'][:, 1], '.', color=c, markersize=3)
    ax1.set_title(f"BEFORE: {total_pts_before} points")
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    for p in redistributed:
        cat = p.get('category', classify_polyline_category(p))
        c = '#ff0000' if cat == 'boundary' else '#000000'
        ax2.plot(p['coords'][:, 0], p['coords'][:, 1], color=c, linewidth=1.0)
        ax2.plot(p['coords'][:, 0], p['coords'][:, 1], '.', color=c, markersize=3)
    fc = params['focus_center']
    r_min = params['r_min_dist']
    r_max = params['r_max_dist']
    ax2.add_patch(Circle(fc, r_min, color='orange', fill=True, alpha=0.10, ec='orange', linestyle='--'))
    ax2.add_patch(Circle(fc, r_max, color='red', fill=False, linestyle=':', alpha=0.4))
    ax2.plot(fc[0], fc[1], 'o', color='orange', markersize=8)
    ax2.set_title(f"AFTER: {total_pts_after} points | min={params['min_spacing']} max={params['max_spacing']}")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    compare_path = "before_after_comparison.png"
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison plot: {compare_path}")
    plt.show()

if __name__ == "__main__":
    main()
