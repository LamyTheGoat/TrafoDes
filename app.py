import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time

# ==============================================================================
# 1. 3D GÖRSELLEŞTİRME MOTORU (V5 - ZERO GAP & SOLID FIT)
# ==============================================================================

def get_cuboid_mesh(center, size, color, name):
    x, y, z = center
    dx, dy, dz = size
    x_p = [x-dx/2, x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2, x+dx/2, x+dx/2]
    y_p = [y-dy/2, y+dy/2, y+dy/2, y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2]
    z_p = [z-dz/2, z-dz/2, z-dz/2, z-dz/2, z+dz/2, z+dz/2, z+dz/2, z+dz/2]
    
    return go.Mesh3d(
        x=x_p, y=y_p, z=z_p,
        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color, opacity=1.0, name=name, flatshading=True, lighting=dict(ambient=0.7, diffuse=0.8)
    )

def get_solid_coil_layer(center, r_in, r_out, height, z_bottom, flat_len, color, name, opacity=1.0):
    """
    Bobinleri iç içe geçmiş katı bloklar olarak çizer. Boşluk bırakmaz.
    """
    xc, yc = center
    
    # Çözünürlük
    theta = np.linspace(0, 2*np.pi, 72) # Daha yuvarlak
    z = np.linspace(z_bottom, z_bottom + height, 2) # Sadece alt ve üst
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_circ = np.cos(theta_grid)
    y_circ = np.sin(theta_grid)
    
    # Obround (Yassılaşma)
    y_shift = np.zeros_like(y_circ)
    if flat_len > 0:
        y_shift[y_circ > 0] = flat_len / 2
        y_shift[y_circ < 0] = -flat_len / 2
    
    # --- YÜZEYLER ---
    traces = []
    
    # 1. Dış Duvar
    x_outer = (r_out * x_circ) + xc
    y_outer = (r_out * y_circ) + y_shift + yc
    traces.append(go.Surface(x=x_outer, y=y_outer, z=z_grid, 
                             colorscale=[[0, color], [1, color]], 
                             opacity=opacity, showscale=False, name=f'{name} Out'))
    
    # 2. Üst Kapak (Görsel bütünlük için)
    # Üst yüzeyi kapatmak için r_in'den r_out'a tarama yapıyoruz
    r_range = np.linspace(r_in, r_out, 4)
    theta_cap, r_cap = np.meshgrid(theta, r_range)
    
    x_c = np.cos(theta_cap)
    y_c = np.sin(theta_cap)
    
    y_s_cap = np.zeros_like(y_c)
    if flat_len > 0:
        y_s_cap[y_c > 0] = flat_len / 2
        y_s_cap[y_c < 0] = -flat_len / 2

    x_lid = (r_cap * x_c) + xc
    y_lid = (r_cap * y_c) + y_s_cap + yc
    z_lid = np.full_like(x_lid, z_bottom + height)
    
    traces.append(go.Surface(x=x_lid, y=y_lid, z=z_lid,
                             colorscale=[[0, color], [1, color]],
                             opacity=opacity, showscale=False, name=f'{name} Top'))

    return traces

def draw_3d_transformer(dims):
    fig = go.Figure()
    
    cd = dims['core_dia']
    fl = dims.get('flat_len', 0)
    fh = dims['foil_height']
    
    # --- Z EKSENİ HESAPLAMALARI (SIFIR BOŞLUK) ---
    # Referans: Z=0 en alt nokta
    
    yoke_height = cd # Boyunduruk yüksekliği genelde bacak çapı kadardır veya yakındır
    
    z_bot_yoke_start = 0
    z_bot_yoke_end = z_bot_yoke_start + yoke_height
    
    # Bacak ve Bobinler tam olarak boyunduruğun bittiği yerden başlar
    z_active_part_start = z_bot_yoke_end
    z_active_part_end = z_active_part_start + fh + 20 # +20mm boyunduruk izolasyonu (minimum)
    
    # Bobin Yüksekliği (Pencereye tam sığsın)
    coil_height = fh
    z_coil_start = z_active_part_start + 10 # Alttan 10mm takoz
    # Üstten de 10mm kalır, toplam pencere = fh + 20
    
    z_top_yoke_start = z_active_part_end
    z_top_yoke_end = z_top_yoke_start + yoke_height
    
    # --- RADYAL HESAPLAMALAR (SIFIR BOŞLUK) ---
    # Bobin et kalınlıkları (Görsel)
    lv_build = 30
    hv_build = 40
    channel_gap = 5 # LV-HV arası soğutma kanalı (Önceki 15 idi, düşürdük)
    
    # Bacak Merkezleri Arası Mesafe
    # (CoreDia/2) + LV + Gap + HV + (HvDia/2) + FazArasıBoşluk
    max_radius = (cd/2) + lv_build + channel_gap + hv_build
    leg_spacing = (max_radius * 2) + 15 # Fazlar arası 15mm hava
    total_width = (leg_spacing * 2) + cd
    
    stack_depth = cd + fl
    
    c_core = '#2C3E50' # Metalik Gri
    c_lv = '#D35400'   # Bakır
    c_hv = '#27AE60'   # Yeşil
    
    # 1. NÜVE PARÇALARI
    # Merkez koordinatlarını hesapla
    center_bot_z = (z_bot_yoke_start + z_bot_yoke_end) / 2
    center_top_z = (z_top_yoke_start + z_top_yoke_end) / 2
    center_leg_z = (z_active_part_start + z_active_part_end) / 2
    leg_h = z_active_part_end - z_active_part_start
    
    # Alt Boyunduruk
    fig.add_trace(get_cuboid_mesh((0, 0, center_bot_z), (total_width, stack_depth, yoke_height), c_core, 'Bottom Yoke'))
    # Üst Boyunduruk
    fig.add_trace(get_cuboid_mesh((0, 0, center_top_z), (total_width, stack_depth, yoke_height), c_core, 'Top Yoke'))
    
    for x_c in [-leg_spacing, 0, leg_spacing]:
        # Bacak (Dikdörtgen Prizma)
        fig.add_trace(get_cuboid_mesh((x_c, 0, center_leg_z), (cd, stack_depth, leg_h), c_core, 'Leg'))
        
        # 2. BOBİNLER
        
        # LV Sargısı (Nüveye tam yapışık başlasın: +1mm)
        r_lv_in = (cd/2) + 1 
        r_lv_out = r_lv_in + lv_build
        
        lv_traces = get_solid_coil_layer(
            (x_c, 0), r_lv_in, r_lv_out, coil_height, z_coil_start, fl, c_lv, "LV", opacity=1.0
        )
        for t in lv_traces: fig.add_trace(t)
        
        # HV Sargısı
        r_hv_in = r_lv_out + channel_gap
        r_hv_out = r_hv_in + hv_build
        
        hv_traces = get_solid_coil_layer(
            (x_c, 0), r_hv_in, r_hv_out, coil_height - 20, z_coil_start + 10, fl, c_hv, "HV", opacity=0.75
        )
        for t in hv_traces: fig.add_trace(t)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False), 
            aspectmode='data' # Orantıyı koru
        ),
        margin=dict(l=0, r=0, b=0, t=0), 
        height=550
    )
    return fig
# ==============================================================================
# 2. SİMÜLASYON MOTORU (AYNI KALDI)
# ==============================================================================
def mock_optimization_engine(inputs):
    time.sleep(0.4)
    power = inputs['power']
    is_copper = (inputs['mat_hv'] == 'Copper')
    is_obround = inputs.get('is_obround', False)
    use_stock = inputs.get('use_stock', False)
    
    base_cost = power * 18 
    if is_copper: base_cost *= 1.45
    
    flat_len = 0
    if is_obround:
        base_cost *= 1.06
        flat_len = 80 
    
    final_foil_h = 550
    if use_stock:
        stock_h = inputs.get('stock_foil_h', 600)
        final_foil_h = stock_h
        if stock_h > 550: base_cost *= 1.12
    
    # Tolerans Limitleri
    guar_nll = inputs.get('guar_nll', 300)
    limit_nll = guar_nll * (1 + inputs.get('tol_nll', 0)/100)
    guar_ll = inputs.get('guar_ll', 3000)
    limit_ll = guar_ll * (1 + inputs.get('tol_ll', 0)/100)

    best_design = {
        "price": base_cost,
        "weight": power * 3.4,
        "losses_nll": guar_nll * 0.95,
        "losses_ll": guar_ll * 0.98,
        "ucc": inputs.get('target_ucc', 4.0) + 0.15,
        "dimensions": {
            "core_dia": 160,
            "foil_height": final_foil_h,
            "flat_len": flat_len,
            "core_len": 160 + flat_len
        },
        "limits": {"nll": limit_nll, "ll": limit_ll},
        "bom": {"Core Steel": base_cost*0.32, "Conductors": base_cost*0.48, "Insulation": base_cost*0.1, "Labor": base_cost*0.1}
    }
    
    second_best = {"price": base_cost * 1.045, "diff_percent": 4.5}
    market_avg = {"price": base_cost * 1.15, "diff_percent": 15.0}
    
    scatter_data = pd.DataFrame({
        "Core Diameter (mm)": np.random.normal(160, 25, 120),
        "Price ($)": np.random.normal(base_cost * 1.3, base_cost * 0.15, 120)
    })
    scatter_data = pd.concat([scatter_data, pd.DataFrame({"Core Diameter (mm)": [160], "Price ($)": [base_cost]})], ignore_index=True)
    
    return {"best": best_design, "second": second_best, "market": market_avg, "scatter": scatter_data}

# ==============================================================================
# 3. GRAFİKLER (AYNI KALDI)
# ==============================================================================
def plot_losses_gauge(val, limit, title):
    if limit is None or limit == 0: limit = val * 1.1
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = val,
        delta = {'reference': limit, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        title = {'text': title, 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [None, limit*1.3]},
            'bar': {'color': "#2ECC71" if val <= limit else "#E74C3C"},
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': limit}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20))
    return fig

def plot_design_space(scatter_data, best_price):
    fig = px.scatter(scatter_data, x="Core Diameter (mm)", y="Price ($)", color="Price ($)", color_continuous_scale="RdYlGn_r", opacity=0.6)
    fig.add_trace(go.Scatter(x=[160], y=[best_price], mode='markers', marker=dict(color='red', size=18, symbol='star', line=dict(width=2, color='black')), name='Winner'))
    fig.update_layout(height=350, margin=dict(t=40, b=0), xaxis_title="Core Diameter", yaxis_title="Price")
    return fig

# ==============================================================================
# 4. STREAMLIT UI (ULTIMATE V4)
# ==============================================================================
st.set_page_config(page_title="Altinsoy Ultimate Designer V4", layout="wide", page_icon="⚡")

with st.sidebar:
    st.image("https://via.placeholder.com/200x50?text=ALTINSOY+ENERJI", use_container_width=True)
    mode = st.radio("Mode", ["Sales 💼", "Engineer 🛠️"], horizontal=True)
    
    st.markdown("### 1. Core Specs")
    power = st.number_input("Power (kVA)", 50, 5000, 250, step=50)
    voltage = st.selectbox("HV Voltage", ["11 kV", "33 kV", "34.5 kV"])
    is_obround = st.checkbox("Obround Core (Stadium)", value=False)
    
    st.markdown("### 2. Materials & Stock")
    c1, c2 = st.columns(2)
    mat_hv = c1.selectbox("HV", ["Al", "Cu"])
    mat_lv = c2.selectbox("LV", ["Al", "Cu"])
    
    use_stock = st.checkbox("Use Stock Inventory")
    stock_inputs = {}
    if use_stock:
        st.info("Inventory Active")
        sc1, sc2 = st.columns(2)
        stock_inputs['stock_foil_h'] = sc1.number_input("Foil W (mm)", value=600)
        stock_inputs['stock_wire_d'] = sc2.number_input("Wire D (mm)", value=2.5)

    eng_inputs = {}
    if mode == "Engineer 🛠️":
        st.markdown("### 3. Tolerances")
        with st.expander("Loss Limits"):
            ec1, ec2 = st.columns(2)
            eng_inputs['guar_nll'] = ec1.number_input("Po (W)", value=300)
            eng_inputs['tol_nll'] = ec2.number_input("Tol Po (%)", value=0.0)
            ec3, ec4 = st.columns(2)
            eng_inputs['guar_ll'] = ec3.number_input("Pk (W)", value=3250)
            eng_inputs['tol_ll'] = ec4.number_input("Tol Pk (%)", value=0.0)
            eng_inputs['target_ucc'] = st.number_input("Uk (%)", value=4.0)
    else:
        eng_inputs = {'guar_nll': 300, 'tol_nll': 0, 'guar_ll': 3250, 'tol_ll': 10, 'target_ucc': 4.0}

    st.markdown("---")
    run_btn = st.button("🚀 OPTIMIZE", type="primary", use_container_width=True)

st.title("⚡ Transformer Design Optimization Suite")

if run_btn:
    all_inputs = {"power": power, "mat_hv": "Copper" if mat_hv=="Cu" else "Aluminum", 
                  "mat_lv": "Copper" if mat_lv=="Cu" else "Aluminum", 
                  "is_obround": is_obround, "use_stock": use_stock}
    all_inputs.update(stock_inputs)
    all_inputs.update(eng_inputs)
    
    with st.spinner("Processing..."):
        res = mock_optimization_engine(all_inputs)
        best = res['best']
        dims = best['dimensions']
        
        # KIYASLAMA
        st.markdown("#### 🏆 Market Position")
        cols = st.columns(3)
        cols[0].metric("WINNER PRICE", f"${best['price']:,.0f}", "Global Optimum")
        cols[1].metric("vs 2nd Best", f"${res['second']['price']:,.0f}", f"+{res['second']['diff_percent']}% Expensive", delta_color="inverse")
        cols[2].metric("vs Market Avg", f"${res['market']['price']:,.0f}", f"+{res['market']['diff_percent']}% Expensive", delta_color="inverse")
        st.divider()
        
        # TABS
        tab_3d, tab_perf, tab_space, tab_cost = st.tabs(["🧊 3D Model", "📊 Performance", "🎯 Design Space", "💰 Costing"])
        
        with tab_3d:
            c3d1, c3d2 = st.columns([3, 1])
            with c3d1:
                st.plotly_chart(draw_3d_transformer(dims), use_container_width=True)
            with c3d2:
                st.success(f"Shape: {'Stadium' if is_obround else 'Circular'}")
                st.table(pd.DataFrame({"Param": ["Core Dia", "Foil Height", "Stack Depth"], "Value": [dims['core_dia'], dims['foil_height'], dims['core_len']]}))

        with tab_perf:
            gp1, gp2 = st.columns(2)
            with gp1: st.plotly_chart(plot_losses_gauge(best['losses_nll'], best['limits']['nll'], "No-Load Loss (Po)"), use_container_width=True)
            with gp2: st.plotly_chart(plot_losses_gauge(best['losses_ll'], best['limits']['ll'], "Load Loss (Pk)"), use_container_width=True)
            st.write(f"**Calculated Uk:** %{best['ucc']:.2f}")

        with tab_space:
            st.plotly_chart(plot_design_space(res['scatter'], best['price']), use_container_width=True)
            
        with tab_cost:
            bom = pd.DataFrame(list(best['bom'].items()), columns=["Item", "Cost"])
            col_b1, col_b2 = st.columns(2)
            col_b1.plotly_chart(px.pie(bom, values="Cost", names="Item", hole=0.4), use_container_width=True)
            col_b2.dataframe(bom, use_container_width=True)

else:
    st.info("👈 Set parameters and click OPTIMIZE.")