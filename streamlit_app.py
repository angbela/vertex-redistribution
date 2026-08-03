import os
import io
import json
import uuid
import tempfile
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mesh_density import (
    extract_polylines, split_domain_polyline, classify_polyline_category,
    redistribute_polyline, snap_and_pin_endpoints, export_xyz, export_dxf,
)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LAYERS = ['domain', 'Batas Pantai', 'batas pantai', 'Daratan', 'Kontur']

PAGE_CONFIG = {
    'page_title': 'Mesh Density Redistributor',
    'page_icon': '🌊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'original_filename' not in st.session_state:
    st.session_state.original_filename = None
if 'polylines' not in st.session_state:
    st.session_state.polylines = None
if 'geojson_before' not in st.session_state:
    st.session_state.geojson_before = None
if 'bbox' not in st.session_state:
    st.session_state.bbox = None
if 'span_x' not in st.session_state:
    st.session_state.span_x = None
if 'span_y' not in st.session_state:
    st.session_state.span_y = None
if 'params' not in st.session_state:
    st.session_state.params = None
if 'redistributed' not in st.session_state:
    st.session_state.redistributed = None
if 'total_before' not in st.session_state:
    st.session_state.total_before = None
if 'total_after' not in st.session_state:
    st.session_state.total_after = None


def step_indicator(current_step):
    steps = ['Upload DXF', 'Define Focus', 'Process', 'Download']
    cols = st.columns(4)
    for i, (col, name) in enumerate(zip(cols, steps)):
        n = i + 1
        if n < current_step:
            col.success(f'✅ **{n}. {name}**')
        elif n == current_step:
            col.info(f'📍 **{n}. {name}**')
        else:
            col.markdown(f'<span style="color:#9ca3af"><b>{n}.</b> {name}</span>', unsafe_allow_html=True)
    st.markdown('---')


def job_dir(job_id):
    d = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def load_and_process_dxf(filepath):
    polylines = extract_polylines(filepath, target_layers=TARGET_LAYERS)
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
    return all_polys


def polylines_to_geojson(polylines):
    xs = []
    ys = []
    features = []
    for i, p in enumerate(polylines):
        cat = p.get('category', classify_polyline_category(p))
        if cat == 'boundary':
            stroke = '#ff0000'
            width = 3
        elif cat == 'coastline':
            stroke = '#000000'
            width = 2
        else:
            stroke = '#444488'
            width = 1.5
        coords = p['coords']
        xs.extend(coords[:, 0].tolist())
        ys.extend(coords[:, 1].tolist())
        features.append({
            'type': 'Feature',
            'properties': {
                'id': i,
                'layer': p['layer'],
                'category': cat,
                'stroke': stroke,
                'stroke-width': width,
                'n_pts': len(coords),
            },
            'geometry': {
                'type': 'LineString',
                'coordinates': coords[:, :2].tolist(),
            }
        })
    fc = {
        'type': 'FeatureCollection',
        'features': features,
        'bbox': [min(xs), min(ys), max(xs), max(ys)],
    }
    return fc


def plot_geometry(polylines, bbox=None, focus_center=None, r_min=None, r_max=None, title=None, show_dots=True):
    all_x = np.concatenate([p['coords'][:, 0] for p in polylines]) if polylines else np.array([])
    all_y = np.concatenate([p['coords'][:, 1] for p in polylines]) if polylines else np.array([])
    if bbox is None and len(all_x) > 0:
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
    elif bbox is not None:
        xmin, ymin, xmax, ymax = bbox
    else:
        xmin, xmax, ymin, ymax = 0, 1, 0, 1

    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)

    if focus_center is not None:
        r_pad = 0
        if r_max is not None and r_max > 0:
            r_pad = r_max * 1.08
        elif r_min is not None and r_min > 0:
            r_pad = r_min * 2.0
        fc_x, fc_y = focus_center
        xmin = min(xmin, fc_x - r_pad) if r_pad else xmin
        xmax = max(xmax, fc_x + r_pad) if r_pad else xmax
        ymin = min(ymin, fc_y - r_pad) if r_pad else ymin
        ymax = max(ymax, fc_y + r_pad) if r_pad else ymax
        span_x = max(xmax - xmin, 1e-6)
        span_y = max(ymax - ymin, 1e-6)

    pad_x = span_x * 0.05
    pad_y = span_y * 0.05
    lim_x = (xmin - pad_x, xmax + pad_x)
    lim_y = (ymin - pad_y, ymax + pad_y)

    aspect = span_x / span_y
    fig_w = 11.0
    fig_h = fig_w / max(min(aspect, 2.5), 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 9.0)))

    for p in polylines:
        cat = p.get('category', 'other')
        c = '#ff0000' if cat == 'boundary' else ('#000000' if cat == 'coastline' else '#444488')
        lw = 2.2 if cat == 'boundary' else (1.5 if cat == 'coastline' else 1.2)
        ax.plot(p['coords'][:, 0], p['coords'][:, 1], color=c, linewidth=lw, solid_capstyle='round')
        if show_dots:
            ax.plot(p['coords'][:, 0], p['coords'][:, 1], '.', color=c, markersize=2.5, zorder=3)

    if focus_center is not None:
        fc = focus_center
        diag = np.sqrt(span_x ** 2 + span_y ** 2)
        if r_max is not None and r_max > 0:
            ax.add_patch(Circle((fc[0], fc[1]), r_max, color='#cc0000',
                                fill=False, ls=':', alpha=0.55, lw=1.4, zorder=4))
            ax.add_patch(Circle((fc[0], fc[1]), r_max, color='red',
                                fill=True, alpha=0.03, zorder=1))
        if r_min is not None and r_min > 0:
            ax.add_patch(Circle((fc[0], fc[1]), r_min, color='#ff9900',
                                fill=True, alpha=0.12, ec='#ff9900', ls='--', lw=1.6, zorder=4))
        ax.plot(fc[0], fc[1], 'o', color='#ffcc00', markersize=12,
                markeredgecolor='white', markeredgewidth=1.8, zorder=6)

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    try:
        ax.set_aspect('equal', adjustable='box')
    except Exception:
        pass
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.28, linewidth=0.6)
    ax.tick_params(axis='both', labelsize=8.5)
    plt.tight_layout(pad=1.2)
    return fig


def plot_before_after(before, after, params, total_before, total_after):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax1, ax2 = axes
    for p in before:
        cat = p.get('category', 'other')
        c = '#ff0000' if cat == 'boundary' else '#000000'
        ax1.plot(p['coords'][:, 0], p['coords'][:, 1], color=c, linewidth=0.9)
        ax1.plot(p['coords'][:, 0], p['coords'][:, 1], '.', color=c, markersize=2.5)
    ax1.set_title(f'BEFORE: {total_before:,} points', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    for p in after:
        cat = p.get('category', 'other')
        c = '#ff0000' if cat == 'boundary' else '#000000'
        ax2.plot(p['coords'][:, 0], p['coords'][:, 1], color=c, linewidth=0.9)
        ax2.plot(p['coords'][:, 0], p['coords'][:, 1], '.', color=c, markersize=2.5)
    fc = params['focus_center']
    ax2.add_patch(Circle(fc, params['r_min_dist'], color='orange',
                         fill=True, alpha=0.10, ec='orange', ls='--', lw=1.2))
    ax2.add_patch(Circle(fc, params['r_max_dist'], color='red',
                         fill=False, ls=':', alpha=0.4, lw=1.2))
    ax2.plot(fc[0], fc[1], 'o', color='orange', markersize=9,
             markeredgecolor='white', markeredgewidth=1.2)
    ax2.set_title(f'AFTER: {total_after:,} points | min={params["min_spacing"]:.2f} max={params["max_spacing"]:.2f}',
                  fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def legend_markdown():
    st.markdown('''
    **Legend:**
    - 🔴 <span style="color:#dc2626">**Red lines**</span> — Boundary (domain edges)
    - ⚫ <span style="color:#111">**Black lines**</span> — Coastline
    - 🔵 <span style="color:#444488">**Blue/indigo lines**</span> — Contour / Other
    - 🟡 **Yellow dot** — Focus center (port zone)
    - 🟠 **Orange dashed circle** — Min-spacing radius
    - 🔴 **Red dotted circle** — Outer / max-spacing boundary
    ''', unsafe_allow_html=True)


def step1_upload():
    st.subheader('📐 Step 1 — Upload DXF File')
    st.markdown('_Coastline + boundary layers (Batas Pantai, domain, Kontur, Daratan)_')
    uploaded = st.file_uploader('Choose a .dxf file', type=['dxf'], accept_multiple_files=False,
                                help='DXF file containing coastline and boundary polylines.')
    if uploaded is not None:
        with st.spinner('Parsing DXF file...'):
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                polylines = load_and_process_dxf(tmp_path)
            except Exception as e:
                st.error(f'❌ Failed to read DXF: {e}')
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if len(polylines) == 0:
            st.warning('⚠️ No polylines found in target layers. Expected layers: ' +
                       ', '.join(TARGET_LAYERS))
            return

        job_id = uuid.uuid4().hex[:10]
        st.session_state.job_id = job_id
        st.session_state.original_filename = uploaded.name
        st.session_state.polylines = polylines
        geojson = polylines_to_geojson(polylines)
        st.session_state.geojson_before = geojson
        bbox = geojson['bbox']
        st.session_state.bbox = bbox
        span_x = bbox[2] - bbox[0]
        span_y = bbox[3] - bbox[1]
        st.session_state.span_x = span_x
        st.session_state.span_y = span_y
        save_dir = job_dir(job_id)
        with open(os.path.join(save_dir, 'polylines_before.geojson'), 'w') as fh:
            json.dump(geojson, fh)

        total_segments = len(geojson['features'])
        total_pts = sum(f['properties']['n_pts'] for f in geojson['features'])

        st.success(f'✅ Successfully loaded **{uploaded.name}**')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Segments', f'{total_segments}')
        c2.metric('Total Points', f'{total_pts:,}')
        c3.metric('X Span', f'{span_x:.2f}')
        c4.metric('Y Span', f'{span_y:.2f}')

        with st.expander('🗺️ Geometry Preview', expanded=True):
            fig = plot_geometry(polylines, bbox=bbox, title='Uploaded Geometry (before redistribution)')
            st.pyplot(fig, clear_figure=True)
            legend_markdown()

        st.markdown('---')
        if st.button('➡️ Proceed to Step 2 — Define Focus Area', use_container_width=True, type='primary'):
            st.session_state.step = 2
            st.rerun()


def step2_focus():
    st.subheader('🎯 Step 2 — Define Focus Area & Parameters')
    st.markdown(f'📎 **File:** `{st.session_state.original_filename}` · **Job:** `{st.session_state.job_id}`')
    polylines = st.session_state.polylines
    bbox = st.session_state.bbox
    span_x = st.session_state.span_x
    span_y = st.session_state.span_y
    xmin, ymin, xmax, ymax = bbox

    total_segments = len(polylines)
    total_pts = sum(len(p['coords']) for p in polylines)

    left, right = st.columns([1.3, 1])

    with right:
        st.markdown('#### 📊 Stats')
        sc1, sc2 = st.columns(2)
        sc1.metric('Segments', f'{total_segments}')
        sc2.metric('Points', f'{total_pts:,}')
        sc3, sc4 = st.columns(2)
        sc3.metric('X span', f'{span_x:.2f}')
        sc4.metric('Y span', f'{span_y:.2f}')

        st.markdown('#### 📍 Focus Center (port zone)')
        default_fx = xmin + span_x * 0.35
        default_fy = ymin + span_y * 0.55
        fx = st.number_input('Focus X', value=float(default_fx),
                             min_value=float(xmin), max_value=float(xmax),
                             step=max(1.0, span_x / 500.0), format='%.4f')
        fy = st.number_input('Focus Y', value=float(default_fy),
                             min_value=float(ymin), max_value=float(ymax),
                             step=max(1.0, span_y / 500.0), format='%.4f')
        focus_center = np.array([fx, fy], dtype=np.float64)

        st.markdown('#### 🎯 Radii')
        default_r_min = min(span_x, span_y) * 0.10
        default_r_max = float(np.sqrt(span_x ** 2 + span_y ** 2) * 0.50)
        r_min = st.slider('Min-spacing radius (orange zone edge)',
                          min_value=0.0, max_value=float(max(span_x, span_y)),
                          value=float(default_r_min), step=max(0.1, default_r_min / 100.0))
        r_max = st.slider('Outer boundary radius (red zone edge)',
                          min_value=max(0.1, r_min * 1.01),
                          max_value=float(max(span_x, span_y) * 1.5),
                          value=float(default_r_max),
                          step=max(0.1, default_r_max / 100.0))

        st.markdown('#### ⚙️ Spacing Parameters')
        default_min_spacing = round(max(1.0, min(span_x, span_y) * 0.005), 1)
        default_max_spacing = round(max(1.0, min(span_x, span_y) * 0.05), 1)
        min_spacing = st.number_input('Min spacing (in focus zone)',
                                      min_value=0.01, value=float(default_min_spacing),
                                      step=0.5, format='%.2f')
        max_spacing = st.number_input('Max spacing (at boundary)',
                                      min_value=max(0.01, min_spacing),
                                      value=max(float(default_max_spacing), float(min_spacing)),
                                      step=0.5, format='%.2f')

        params = {
            'focus_center': focus_center,
            'r_min_dist': float(r_min),
            'r_max_dist': float(r_max),
            'min_spacing': float(min_spacing),
            'max_spacing': float(max_spacing),
        }
        st.session_state.params = params

        st.markdown('#### 🧾 Summary')
        with st.container(border=True):
            st.write(f'**Center:** ({fx:.2f}, {fy:.2f})')
            st.write(f'**R min:** {r_min:.2f}  ·  **R max:** {r_max:.2f}')
            st.write(f'**Spacing:** {min_spacing:.2f} → {max_spacing:.2f}')

        ok = r_min > 0 and r_max > r_min and min_spacing > 0 and max_spacing >= min_spacing
        if not ok:
            st.warning('⚠️ Parameters invalid — need: 0 < R_min < R_max and 0 < min_spacing ≤ max_spacing')

        col1, col2 = st.columns(2)
        with col1:
            if st.button('⬅️ Back to Upload', use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button('▶️ Run Redistribution', disabled=not ok,
                         use_container_width=True, type='primary'):
                st.session_state.step = 3
                st.rerun()

    with left:
        st.markdown('#### 🗺️ Interactive Preview')
        fig = plot_geometry(polylines, bbox=bbox,
                            focus_center=focus_center,
                            r_min=r_min if r_min > 0 else None,
                            r_max=r_max if r_max > 0 else None,
                            title='Geometry — adjust sliders on right to update preview')
        st.pyplot(fig, clear_figure=True)
        legend_markdown()


def step3_process():
    st.subheader('⏳ Step 3 — Running Redistribution')
    progress_bar = st.progress(0, text='Preparing...')

    polylines = st.session_state.polylines
    params = st.session_state.params
    job_id = st.session_state.job_id

    progress_bar.progress(5, text='Rebuilding polylines from GeoJSON...')

    polylines2 = []
    for feat in st.session_state.geojson_before['features']:
        coords = np.array(feat['geometry']['coordinates'], dtype=np.float64)
        props = feat['properties']
        polylines2.append({
            'layer': props.get('layer', 'unknown'),
            'color_idx': 1 if props.get('category') == 'boundary' else 7,
            'color': props.get('stroke', '#000000'),
            'coords': coords,
            'category': props.get('category', 'other'),
        })

    total_before = sum(len(p['coords']) for p in polylines2)
    st.session_state.total_before = total_before

    progress_bar.progress(15, text=f'Redistributing {len(polylines2)} segments...')

    redistributed = []
    total_after = 0
    N = len(polylines2)
    for i, p in enumerate(polylines2):
        new_coords = redistribute_polyline(p['coords'], params)
        p_new = dict(p)
        p_new['coords'] = new_coords
        total_after += len(new_coords)
        redistributed.append(p_new)
        pct = 15 + int((i + 1) / N * 65)
        progress_bar.progress(pct, text=f'Processing segment {i + 1}/{N}...')

    progress_bar.progress(82, text='Snapping & pinning endpoints...')
    redistributed = snap_and_pin_endpoints(redistributed)

    save_dir = job_dir(job_id)

    progress_bar.progress(87, text='Exporting XYZ scatter file...')
    xyz_path = os.path.join(save_dir, 'redistributed_points.xyz')
    export_xyz(redistributed, xyz_path)

    progress_bar.progress(91, text='Exporting DXF file...')
    dxf_path = os.path.join(save_dir, 'redistributed.dxf')
    export_dxf(redistributed, dxf_path)

    progress_bar.progress(94, text='Saving GeoJSON after...')
    geojson_after = polylines_to_geojson(redistributed)
    with open(os.path.join(save_dir, 'polylines_after.geojson'), 'w') as fh:
        json.dump(geojson_after, fh)

    progress_bar.progress(97, text='Generating before/after comparison plot...')
    fig_ba = plot_before_after(polylines2, redistributed, params, total_before, total_after)
    png_path = os.path.join(save_dir, 'before_after_comparison.png')
    fig_ba.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ba)

    progress_bar.progress(99, text='Saving parameters...')
    params_serializable = {
        'focus_center': [float(params['focus_center'][0]), float(params['focus_center'][1])],
        'r_min_dist': float(params['r_min_dist']),
        'r_max_dist': float(params['r_max_dist']),
        'min_spacing': float(params['min_spacing']),
        'max_spacing': float(params['max_spacing']),
    }
    with open(os.path.join(save_dir, 'params.json'), 'w') as fh:
        json.dump(params_serializable, fh)

    st.session_state.redistributed = redistributed
    st.session_state.total_after = total_after
    progress_bar.progress(100, text='✅ Done!')

    st.success(f'🎉 Redistribution complete! '
               f'**{total_before:,}** → **{total_after:,}** points '
               f'({total_after / total_before:.2f}x)')
    st.session_state.step = 4
    st.rerun()


def step4_results():
    st.subheader('✅ Step 4 — Results & Downloads')
    job_id = st.session_state.job_id
    params = st.session_state.params
    save_dir = job_dir(job_id)
    xyz_path = os.path.join(save_dir, 'redistributed_points.xyz')
    dxf_path = os.path.join(save_dir, 'redistributed.dxf')
    png_path = os.path.join(save_dir, 'before_after_comparison.png')

    st.markdown(f'📎 **File:** `{st.session_state.original_filename}` · **Job:** `{job_id}`')

    total_before = st.session_state.total_before
    total_after = st.session_state.total_after

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric('Points Before', f'{total_before:,}')
    mc2.metric('Points After', f'{total_after:,}',
               delta=f'{((total_after / total_before - 1) * 100):.1f}%')
    mc3.metric('Segments', f'{len(st.session_state.redistributed)}')
    ratio = total_after / total_before if total_before else 0
    mc4.metric('Density Ratio', f'{ratio:.2f}x')

    tab1, tab2 = st.tabs(['🗺️ Before / After Comparison', '📋 Parameters Used'])

    with tab1:
        if os.path.exists(png_path):
            st.image(png_path, caption='Before (left) vs After (right) — vertex density comparison',
                     use_column_width=True)
        else:
            fig = plot_before_after(
                st.session_state.polylines,
                st.session_state.redistributed,
                params, total_before, total_after
            )
            st.pyplot(fig, clear_figure=True)

    with tab2:
        fc = params['focus_center']
        pc1, pc2 = st.columns(2)
        pc1.markdown(f'''
        | Parameter | Value |
        |-----------|-------|
        | 📍 Focus Center X | `{float(fc[0]):.4f}` |
        | 📍 Focus Center Y | `{float(fc[1]):.4f}` |
        | 🟠 Min-spacing Radius | `{params['r_min_dist']:.4f}` |
        | 🔴 Outer Radius | `{params['r_max_dist']:.4f}` |
        ''')
        pc2.markdown(f'''
        | Parameter | Value |
        |-----------|-------|
        | 🟢 Min Spacing | `{params['min_spacing']:.4f}` |
        | 🔴 Max Spacing | `{params['max_spacing']:.4f}` |
        | 📏 Radius Ratio (r_max/r_min) | `{params['r_max_dist'] / max(params['r_min_dist'], 1e-9):.2f}` |
        | 📏 Spacing Range | `{params['max_spacing'] / max(params['min_spacing'], 1e-9):.2f}x` |
        ''')

    st.markdown('---')
    st.markdown('### 📥 Downloads')

    dc1, dc2, dc3 = st.columns(3)

    with open(xyz_path, 'rb') as f:
        xyz_bytes = f.read()
    with open(dxf_path, 'rb') as f:
        dxf_bytes = f.read()
    with open(png_path, 'rb') as f:
        png_bytes = f.read()

    xyz_kb = round(os.path.getsize(xyz_path) / 1024, 1)
    dxf_kb = round(os.path.getsize(dxf_path) / 1024, 1)
    png_kb = round(os.path.getsize(png_path) / 1024, 1)

    with dc1:
        st.download_button(
            label=f'📊 Download XYZ scatter\n({xyz_kb} KB)',
            data=xyz_bytes,
            file_name='redistributed_points.xyz',
            mime='application/octet-stream',
            use_container_width=True,
        )
        st.caption('Scatter — ready for import')

    with dc2:
        st.download_button(
            label=f'📐 Download DXF\n({dxf_kb} KB)',
            data=dxf_bytes,
            file_name='redistributed.dxf',
            mime='application/dxf',
            use_container_width=True,
        )
        st.caption('DXF — reimport to CAD')

    with dc3:
        st.download_button(
            label=f'🖼️ Download Plot PNG\n({png_kb} KB)',
            data=png_bytes,
            file_name='before_after_comparison.png',
            mime='image/png',
            use_container_width=True,
        )
        st.caption('Side-by-side before/after visualization')

    st.markdown('---')
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button('⬅️ Back to Adjust Parameters', use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with ac2:
        if st.button('🔄 Start Over — New File', use_container_width=True, type='primary'):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.step = 1
            st.rerun()


def main():
    st.set_page_config(**PAGE_CONFIG)
    st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    </style>
    """, unsafe_allow_html=True)

    st.title('🌊 Vertex Density Redistributor')
    st.caption('For coastal models — DXF coastline & boundary → variable-spacing vertex redistribution')
    st.markdown('---')

    step_indicator(st.session_state.step)

    if st.session_state.step == 1:
        step1_upload()
    elif st.session_state.step == 2:
        step2_focus()
    elif st.session_state.step == 3:
        step3_process()
    elif st.session_state.step == 4:
        step4_results()


if __name__ == '__main__':
    main()
