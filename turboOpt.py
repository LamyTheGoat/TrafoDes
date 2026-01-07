import math
import time
import numpy as np
from numba import njit, prange

# --- SABİTLER (Aynı) ---
CONNECTION_D = 1
CONNECTION_Y = 1/math.sqrt(3)
AL_DENSITY = 2.7
CU_DENSITY = 8.9
CORE_DENSITY = 7.65
AL_PRICE_FOIL = 3.52
AL_PRICE_WIRE = 4.88
CU_PRICE_FOIL = 11.55
CU_PRICE_WIRE = 11.05
CORE_PRICE = 3.6
AL_RESISTIVITY = 0.0336
CU_RESISTIVITY = 0.00001724
POWERRATING = 160
HVRATE = 33000 * CONNECTION_D
LVRATE = 400 * CONNECTION_Y
FREQUENCY = 50
GUARANTEED_NLL = 242 * 0.98
GUARANTEED_LL = 1925 * 0.98
GUARANTEED_UCC = 4.48
UCC_TOLERANCE = GUARANTEED_UCC * 0.03
PENALTY_NLL_FACTOR = 60
PENALTY_LL_FACTOR = 10
PENALTY_UCC_FACTOR = 10000
INSULATION_THICKNESS_WIRE = 0.12
LV_INSULATION_THICKNESS = 0.125
HV_INSULATION_THICKNESS = 0.5
MAIN_GAP = 9
DISTANCE_CORE_LV = 2
PHASE_GAP = 12
CORE_FILLING_ROUND = 0.84
CORE_FILLING_RECT = 0.97

# --- GÜNCELLENMİŞ PARALEL ÇEKİRDEK ---
@njit(fastmath=True, parallel=True)
def optimize_transformer_variable_length(
    core_dia_arr, turns_arr, height_arr, thick_arr, hvdia_arr, 
    tolerance, core_len_step
):
    
    n_core = len(core_dia_arr)
    n_turns = len(turns_arr)
    n_height = len(height_arr)
    n_thick = len(thick_arr)
    n_hvdia = len(hvdia_arr)
    
    # Sonuç Matrisi: [Price, Turns, Height, Thick, HVDia, CoreDia, CoreLen]
    # Boyutu 6'dan 7'ye çıkardık
    results = np.full((n_core, 7), 1e18, dtype=np.float64)

    # PARALEL DÖNGÜ (Core Diameter)
    for i in prange(n_core): 
        core_dia = core_dia_arr[i]
        
        # Local variables
        local_best_price = 1e18
        best_t = 0.0
        best_h = 0.0
        best_th = 0.0
        best_hv = 0.0
        best_len = 0.0
        
        # --- CORE LENGTH LOOP ---
        # 0'dan başlayıp Core Diameter'a kadar gider
        core_length = 0.0
        while core_length <= core_dia:
            
            # Pre-calculations for Section
            radius = core_dia / 2.0
            section_round = ((core_dia**2) * math.pi / 400.0) * CORE_FILLING_ROUND
            section_rect = (core_length * core_dia / 100.0) * CORE_FILLING_RECT
            core_section = section_round + section_rect
            
            # --- TURNS LOOP ---
            for j in range(n_turns):
                turns = turns_arr[j]
                
                volts_per_turn = LVRATE / turns
                induction = (volts_per_turn * 10000) / (math.sqrt(2) * math.pi * FREQUENCY * core_section)
                
                # --- PRUNING 1: Induction ---
                if induction > 1.95: continue # Çok yüksek, çalışmaz
                if induction < 1.0: 
                    # İndüksiyon çok düşükse, Core Length'i daha da artırmanın anlamı yok.
                    # Çünkü Length artarsa Section artar, Induction daha da düşer.
                    # Bu iç döngüden çıkmak yetmez, Core Length döngüsünü de etkilemek lazım ama
                    # şimdilik sadece 'continue' diyelim.
                    continue

                # Polynomial calc
                w_per_kg = (1.3498 * induction**6) + (-8.1737 * induction**5) + \
                           (19.884 * induction**4) + (-24.708 * induction**3) + \
                           (16.689 * induction**2) + (-5.5386 * induction) + 0.7462
                
                # --- HEIGHT LOOP ---
                for k in range(n_height):
                    height = height_arr[k]
                    window_height = height + 40
                    
                    # --- THICKNESS LOOP ---
                    for l in range(n_thick):
                        thick = thick_arr[l]
                        
                        lv_radial_thick = turns * thick + ((turns - 1) * LV_INSULATION_THICKNESS)
                        
                        # --- HV WIRE LOOP ---
                        for m in range(n_hvdia):
                            hvdia = hvdia_arr[m]
                            
                            # --- CALCULATIONS ---
                            hv_number_of_turns = turns * HVRATE / LVRATE
                            hv_layer_height = height - 20
                            hv_turns_per_layer = (hv_layer_height / (hvdia + INSULATION_THICKNESS_WIRE)) - 1
                            
                            if hv_turns_per_layer <= 0: continue

                            hv_layer_number = math.ceil(hv_number_of_turns / hv_turns_per_layer)
                            hv_radial_thick = hv_layer_number * hvdia + (hv_layer_number - 1) * HV_INSULATION_THICKNESS
                            
                            radial_total = lv_radial_thick + hv_radial_thick + MAIN_GAP + DISTANCE_CORE_LV
                            center_between_legs = (core_dia + radial_total * 2) + PHASE_GAP
                            
                            # Weight Calc
                            rect_weight = (((3 * window_height) + 2 * (2 * center_between_legs + core_dia)) * (core_dia * core_length / 100) * CORE_DENSITY * CORE_FILLING_RECT) / 1e6
                            square_edge = radius * math.sqrt(math.pi)
                            round_weight = (((3 * (window_height + 10)) + 2 * (2 * center_between_legs + core_dia)) * (square_edge * square_edge / 100) * CORE_DENSITY * CORE_FILLING_ROUND) / 1e6
                            core_weight_final = (rect_weight + round_weight) * 100
                            core_price = core_weight_final * CORE_PRICE

                            # NLL
                            nll = w_per_kg * core_weight_final * 1.2
                            
                            # LL & Price
                            lv_avg_dia = core_dia + (2 * DISTANCE_CORE_LV) + lv_radial_thick + (2 * core_length / math.pi)
                            lv_length_total = lv_avg_dia * math.pi * turns
                            hv_avg_dia = core_dia + 2*DISTANCE_CORE_LV + 2*lv_radial_thick + 2*MAIN_GAP + hv_radial_thick + (2*core_length/math.pi)
                            hv_length_total = hv_avg_dia * math.pi * hv_number_of_turns
                            
                            vol_lv = (lv_length_total * height * thick) / 1e6
                            price_lv = vol_lv * AL_DENSITY * AL_PRICE_FOIL
                            
                            vol_hv = (hv_length_total * math.pi * (hvdia**2)) / (4 * 1e6)
                            price_hv = vol_hv * AL_DENSITY * AL_PRICE_WIRE
                            
                            # Losses Check
                            res_lv = (lv_length_total / 1000) * AL_RESISTIVITY / (height * thick)
                            res_hv = (hv_length_total / 1000) * AL_RESISTIVITY / (math.pi * (hvdia**2) / 4)
                            curr_hv = (POWERRATING * 1000) / (HVRATE * 3)
                            curr_lv = (POWERRATING * 1000) / (LVRATE * 3)
                            ll_loss = (res_lv * (curr_lv**2) + res_hv * (curr_hv**2)) * 3 * 1.16
                            
                            if ll_loss > (GUARANTEED_LL * 1.3): continue

                            # UCC Check
                            reduced_width_hv = hv_radial_thick / 3
                            reduced_width_lv = lv_radial_thick / 3
                            denom = reduced_width_lv + reduced_width_hv + MAIN_GAP
                            if denom == 0: denom = 0.001
                            
                            main_gap_dia = core_dia + DISTANCE_CORE_LV*2 + lv_radial_thick*2 + (2*core_length/math.pi) + MAIN_GAP
                            stray_dia = main_gap_dia + reduced_width_hv - reduced_width_lv + ((reduced_width_hv**2 - reduced_width_lv**2) / denom)
                            
                            ux = (POWERRATING * stray_dia * FREQUENCY * denom) / (1210 * (volts_per_turn**2) * height)
                            ur = ll_loss / (10 * POWERRATING)
                            ucc = math.sqrt(ux**2 + ur**2)
                            
                            # Penalties
                            bare_price = core_price + price_lv + price_hv
                            
                            nll_extra = max(0, nll - GUARANTEED_NLL)
                            ll_extra = max(0, ll_loss - GUARANTEED_LL)
                            ucc_diff = max(0, abs(ucc - GUARANTEED_UCC) - abs(UCC_TOLERANCE))
                            
                            # Tolerans Kontrolü
                            if nll_extra > (GUARANTEED_NLL * tolerance / 100): continue
                            if ll_extra > (GUARANTEED_LL * tolerance / 100): continue
                            if ucc_diff > (GUARANTEED_UCC * tolerance / 100): continue

                            total_penalty = (nll_extra * PENALTY_NLL_FACTOR) + (ll_extra * PENALTY_LL_FACTOR) + (ucc_diff * PENALTY_UCC_FACTOR)
                            total_price = bare_price + total_penalty
                            
                            if total_price < local_best_price:
                                local_best_price = total_price
                                best_t = turns
                                best_h = height
                                best_th = thick
                                best_hv = hvdia
                                best_len = core_length
            
            # Length Step Artırımı
            core_length += core_len_step
        
        # Sonuçları Kaydet
        results[i, 0] = local_best_price
        results[i, 1] = best_t
        results[i, 2] = best_h
        results[i, 3] = best_th
        results[i, 4] = best_hv
        results[i, 5] = core_dia
        results[i, 6] = best_len # Yeni parametre

    return results

import ezdxf
import math

import ezdxf
import math
import datetime

def create_professional_drawing(transformer_data, filename="altinsoy_proje_v2.dxf"):
    # --- VERİLERİ ÇEK ---
    price, turns, foil_height, foil_thick, hv_dia, core_dia, core_len = transformer_data[:7]
    
    # Varsayımlar
    lv_insulation = 0.125
    hv_insulation = 0.5
    main_gap = 9
    distance_core_lv = 2
    
    # Hesaplamalar
    lv_radial_thick = turns * foil_thick + ((turns - 1) * lv_insulation)
    
    # HV Tahmini (Görsel için)
    hv_total_thick = 25 # Görsel dolgunluk için
    coil_outer_radius = (core_dia/2) + distance_core_lv + lv_radial_thick + main_gap + hv_total_thick
    
    window_height = foil_height + 40
    leg_center = coil_outer_radius * 2 + 15 # Fazlar arası mesafe (biraz boşluklu)
    
    # --- DXF BAŞLAT ---
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    # --- KATMAN AYARLARI (Profesyonel Renk Kodları) ---
    doc.layers.add(name="0_CERCEVE", color=7)     # Beyaz (Çerçeve)
    doc.layers.add(name="1_NUVE_DIS", color=7)    # Beyaz (Ana hatlar)
    doc.layers.add(name="2_SARGI_LV", color=1)    # Kırmızı
    doc.layers.add(name="3_SARGI_HV", color=3)    # Yeşil
    doc.layers.add(name="4_OLCULENDIRME", color=2)# Sarı
    doc.layers.add(name="5_EKSEN", color=8, linetype="DASHDOT") # Gri
    doc.layers.add(name="6_TARAMA", color=9)      # Açık Gri (Hatch)

    # =================================================================
    # 1. ANTET VE ÇERÇEVE (TITLE BLOCK)
    # =================================================================
    # A3 Kağıt Boyutu (420x297mm) oranında ama ölçekli büyük bir çerçeve
    frame_w = 3000
    frame_h = 2000
    
    # Dış Çerçeve
    msp.add_lwpolyline([
        (0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h), (0, 0)
    ], dxfattribs={'layer': '0_CERCEVE', 'lineweight': 30})
    
    # Antet Kutusu (Sağ Alt)
    antet_w = 800
    antet_h = 250
    msp.add_lwpolyline([
        (frame_w - antet_w, 0), (frame_w, 0), 
        (frame_w, antet_h), (frame_w - antet_w, antet_h),
        (frame_w - antet_w, 0)
    ], dxfattribs={'layer': '0_CERCEVE'})
    
    # Firma Bilgileri
    msp.add_text("ALTINSOY ENERJI", dxfattribs={'height': 40, 'layer': '0_CERCEVE'}).set_pos((frame_w - antet_w + 20, antet_h - 60))
    msp.add_text(f"Proje: {int(price)}$ Trafo Tasarimi", dxfattribs={'height': 25, 'layer': '0_CERCEVE'}).set_pos((frame_w - antet_w + 20, antet_h - 110))
    msp.add_text(f"Tarih: {datetime.datetime.now().strftime('%Y-%m-%d')}", dxfattribs={'height': 20, 'layer': '0_CERCEVE'}).set_pos((frame_w - antet_w + 20, 30))
    
    # Teknik Özet
    ozet_txt = f"Guc: 250kVA | 34.5/0.4kV\nNuve Capi: {core_dia}mm\nFolyo Yuk.: {foil_height}mm"
    msp.add_mtext(ozet_txt, dxfattribs={'char_height': 20, 'layer': '0_CERCEVE'}).set_location((frame_w - 350, antet_h - 50))

    # =================================================================
    # 2. ÜST GÖRÜNÜŞ (PLAN) - KADEMELİ NÜVE DETAYI
    # =================================================================
    # Çizim Merkezi (Çerçevenin sol üst çeyreği)
    top_center_x = frame_w * 0.3
    top_center_y = frame_h * 0.7
    
    # 3 Fazı Çiz
    offsets = [-leg_center, 0, leg_center]
    
    for off in offsets:
        cx, cy = top_center_x + off, top_center_y
        
        # --- EKSENLER ---
        msp.add_line((cx, cy - coil_outer_radius - 50), (cx, cy + coil_outer_radius + 50), dxfattribs={'layer': '5_EKSEN'})
        msp.add_line((cx - coil_outer_radius - 50, cy), (cx + coil_outer_radius + 50, cy), dxfattribs={'layer': '5_EKSEN'})

        # --- KADEMELİ NÜVE (STEPPED CORE) ---
        # Gerçekçi görünmesi için 5 kademe (step) uyduruyoruz
        # Matematiksel olarak daireye yaklaşan dikdörtgenler
        steps = 5
        for i in range(steps):
            # Basit bir daire yaklaşımı (Sinus/Cosinus ile)
            angle = (90 / (steps + 1)) * (i + 1)
            rad_angle = math.radians(angle)
            
            # Kademe Genişliği ve Derinliği (Yarım)
            w = (core_dia / 2) * math.cos(rad_angle)
            d = (core_dia / 2) * math.sin(rad_angle)
            
            # Dikdörtgen (Polyline)
            # Hem dikey hem yatay simetrik dikdörtgenler atarak "haç" şekli ve basamakları oluşturuyoruz
            # 1. Dikey Paket
            msp.add_lwpolyline([
                (cx - w, cy - d), (cx + w, cy - d),
                (cx + w, cy + d), (cx - w, cy + d)
            ], close=True, dxfattribs={'layer': '1_NUVE_DIS'})
            
            # 2. Yatay Paket (Ters açı)
            msp.add_lwpolyline([
                (cx - d, cy - w), (cx + d, cy - w),
                (cx + d, cy + w), (cx - d, cy + w)
            ], close=True, dxfattribs={'layer': '1_NUVE_DIS'})

        # --- SARGILAR (Renkli Daireler) ---
        # LV
        lv_r_in = (core_dia/2) + distance_core_lv
        lv_r_out = lv_r_in + lv_radial_thick
        
        # LV Hatching (Halka şeklinde tarama zor olduğu için iki daire çiziyoruz)
        msp.add_circle((cx, cy), lv_r_in, dxfattribs={'layer': '2_SARGI_LV'})
        msp.add_circle((cx, cy), lv_r_out, dxfattribs={'layer': '2_SARGI_LV'})
        
        # HV
        hv_r_in = lv_r_out + main_gap
        hv_r_out = hv_r_in + hv_total_thick
        
        msp.add_circle((cx, cy), hv_r_in, dxfattribs={'layer': '3_SARGI_HV'})
        msp.add_circle((cx, cy), hv_r_out, dxfattribs={'layer': '3_SARGI_HV'})

    # Boyunduruk (Yoke) Üst Görünüşü
    # Tüm bacakları kapsayan dikdörtgen
    yoke_w = (leg_center * 2) + core_dia
    yoke_d = core_dia # Derinlik
    
    # Nüvenin sadece arkada kalan kısımları (Görsel temizlik için sargıların altından geçirmeyelim, basit dikdörtgen)
    # Detaylı çizimde 'hidden line' olur ama burada ana hat yeterli.
    
    # =================================================================
    # 3. ÖN GÖRÜNÜŞ (KESİT) - TARAMALI
    # =================================================================
    front_x = frame_w * 0.3
    front_y = frame_h * 0.25 # Aşağıda
    
    total_h = window_height + 2*core_dia
    yoke_h = core_dia
    
    # --- NÜVE KESİTİ ---
    # Alt Boyunduruk
    yoke_bot = msp.add_lwpolyline([
        (front_x - yoke_w/2, front_y),
        (front_x + yoke_w/2, front_y),
        (front_x + yoke_w/2, front_y + yoke_h),
        (front_x - yoke_w/2, front_y + yoke_h)
    ], close=True, dxfattribs={'layer': '1_NUVE_DIS'})
    # Tarama Ekle
    hatch = msp.add_hatch(color=9)
    hatch.paths.add_polyline_path(yoke_bot.get_points())
    hatch.set_pattern_fill('ANSI31', scale=5) # Çelik taraması

    # Üst Boyunduruk
    yoke_top_y = front_y + yoke_h + window_height
    yoke_top = msp.add_lwpolyline([
        (front_x - yoke_w/2, yoke_top_y),
        (front_x + yoke_w/2, yoke_top_y),
        (front_x + yoke_w/2, yoke_top_y + yoke_h),
        (front_x - yoke_w/2, yoke_top_y + yoke_h)
    ], close=True, dxfattribs={'layer': '1_NUVE_DIS'})
    # Tarama Ekle
    hatch_top = msp.add_hatch(color=9)
    hatch_top.paths.add_polyline_path(yoke_top.get_points())
    hatch_top.set_pattern_fill('ANSI31', scale=5)

    # Bacaklar
    for off in offsets:
        cx = front_x + off
        # Bacak Dikdörtgeni
        leg_rect = msp.add_lwpolyline([
            (cx - core_dia/2, front_y + yoke_h),
            (cx + core_dia/2, front_y + yoke_h),
            (cx + core_dia/2, yoke_top_y),
            (cx - core_dia/2, yoke_top_y)
        ], close=True, dxfattribs={'layer': '1_NUVE_DIS'})
        # Bacakları taramayalım, çünkü sargılar üstüne gelecek.
        
        # --- SARGILAR (KESİT) ---
        # LV Sol
        lv_l_in = cx - (core_dia/2) - distance_core_lv
        lv_l_out = lv_l_in - lv_radial_thick
        
        coil_bottom = front_y + yoke_h + 20 # Boyunduruktan biraz yukarıda
        coil_top = coil_bottom + foil_height
        
        # LV Sol Kutu
        msp.add_lwpolyline([
            (lv_l_out, coil_bottom), (lv_l_in, coil_bottom),
            (lv_l_in, coil_top), (lv_l_out, coil_top)
        ], close=True, dxfattribs={'layer': '2_SARGI_LV'})
        
        # LV Sağ Kutu
        lv_r_in = cx + (core_dia/2) + distance_core_lv
        lv_r_out = lv_r_in + lv_radial_thick
        msp.add_lwpolyline([
            (lv_r_in, coil_bottom), (lv_r_out, coil_bottom),
            (lv_r_out, coil_top), (lv_r_in, coil_top)
        ], close=True, dxfattribs={'layer': '2_SARGI_LV'})
        
        # HV (LV'nin dışına)
        hv_l_in = lv_l_out - main_gap
        hv_l_out = hv_l_in - hv_total_thick
        
        # HV Sol Kutu (Tarama Örneği)
        hv_poly = msp.add_lwpolyline([
            (hv_l_out, coil_bottom + 10), (hv_l_in, coil_bottom + 10),
            (hv_l_in, coil_top - 10), (hv_l_out, coil_top - 10)
        ], close=True, dxfattribs={'layer': '3_SARGI_HV'})
        
        # HV Taraması (Çapraz)
        hatch_hv = msp.add_hatch(color=3)
        hatch_hv.paths.add_polyline_path(hv_poly.get_points())
        hatch_hv.set_pattern_fill('ANSI37', scale=3) # Bakır/Alüminyum için farklı desen

        # HV Sağ Kutu
        hv_r_in = lv_r_out + main_gap
        hv_r_out = hv_r_in + hv_total_thick
        msp.add_lwpolyline([
            (hv_r_in, coil_bottom + 10), (hv_r_out, coil_bottom + 10),
            (hv_r_out, coil_top - 10), (hv_r_in, coil_top - 10)
        ], close=True, dxfattribs={'layer': '3_SARGI_HV'})


    # =================================================================
    # 4. ÖLÇÜLENDİRME (DIMENSIONS)
    # =================================================================
    # Toplam Genişlik Ölçüsü (Ön Görünüş)
    dim_style = "EZ_M_100_H25_CM" # Varsayılan stil
    
    # Nüve Ekseni Ölçüsü
    msp.add_linear_dim(
        base=(front_x, front_y - 150),
        p1=(front_x + offsets[0], front_y),
        p2=(front_x + offsets[1], front_y),
        dimstyle=dim_style,
        dxfattribs={'layer': '4_OLCULENDIRME'}
    ).render()
    
    # Toplam Yükseklik
    msp.add_linear_dim(
        base=(front_x + yoke_w/2 + 200, front_y + total_h/2),
        p1=(front_x + yoke_w/2, front_y),
        p2=(front_x + yoke_w/2, front_y + total_h),
        dimstyle=dim_style,
        angle=90,
        dxfattribs={'layer': '4_OLCULENDIRME'}
    ).render()

    doc.saveas(filename)
    print(f"Profesyonel çizim oluşturuldu: {filename}")

# --- TEST ---
# create_professional_drawing(best_transformer)
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import ezdxf

def convert_dxf_to_png(dxf_filename, png_filename):
    # 1. DXF Dosyasını Oku
    doc = ezdxf.readfile(dxf_filename)
    msp = doc.modelspace()

    # 2. Render Ayarları (Siyah Arka Plan, Renkli Çizgiler)
    # Arka planı beyaz yapmak istersen: RenderContext(doc) yeterli
    ctx = RenderContext(doc)
    
    # 3. Çizim Alanı Oluştur
    fig = plt.figure(figsize=(20, 12), dpi=300) # Yüksek çözünürlük
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off() # Eksenleri gizle
    
    # 4. Matplotlib ile Çizdir
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    
    # 5. Kaydet
    fig.savefig(png_filename, dpi=300)
    print(f"Resim oluşturuldu: {png_filename}")
    plt.close(fig) # Belleği temizle


import matplotlib.pyplot as plt
import numpy as np

# --- (Önceki optimize_transformer_precision fonksiyonun çalıştığını ve 'results' döndürdüğünü varsayıyoruz) ---

def visualize_design_space(results):
    print("Grafikler hazırlanıyor...")
    
    # 1. Veri Temizliği: Fiyatı sonsuz (1e18) olan geçersiz tasarımları ele
    # Sadece "Geçerli" (Valid) tasarımları al
    valid_mask = results[:, 0] < 1e17
    valid_data = results[valid_mask]
    
    if len(valid_data) == 0:
        print("Hata: Hiç geçerli tasarım bulunamadı, grafik çizilemiyor.")
        return

    # En iyi (en ucuz) tasarımı bul
    best_idx = np.argmin(valid_data[:, 0])
    best_design = valid_data[best_idx]
    
    # --- GRAFİK 1: NÜVE ÇAPI vs FİYAT (Genel Bakış) ---
    plt.figure(figsize=(12, 6))
    
    # Tüm geçerli noktaları çiz (Mavi, küçük noktalar)
    # alpha=0.3 ile şeffaflık veriyoruz ki yoğun bölgeler daha koyu görünsün
    plt.scatter(valid_data[:, 5], valid_data[:, 0], 
                alpha=0.3, c='blue', s=5, label='Geçerli Tasarımlar')
    
    # En iyi noktayı işaretle (Kırmızı, büyük nokta)
    plt.scatter(best_design[5], best_design[0], 
                color='red', s=100, marker='*', label=f'En İyi: {best_design[0]:.0f}$')
    
    plt.title('Tasarım Uzayı: Nüve Çapına Göre Maliyet Dağılımı')
    plt.xlabel('Nüve Çapı (mm)')
    plt.ylabel('Toplam Maliyet ($)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Y eksenini (Fiyat) biraz kırpalım ki "uçuk" fiyatlar grafiği bozmasın
    # En iyi fiyatın %150'sine kadar olan kısmı gösterelim
    plt.ylim(best_design[0] * 0.9, best_design[0] * 1.5)
    
    # --- GRAFİK 2: ISI HARİTASI (Core Dia vs Foil Height) ---
    # Bu grafik, hangi çap ve yükseklik kombinasyonlarının ucuz olduğunu renklerle gösterir.
    plt.figure(figsize=(12, 7))
    
    # Renklendirme (Color map) fiyata göre olacak
    # 'viridis_r' ters viridis demek (Sarı=Ucuz, Mor=Pahalı olsun diye _r ekledik veya tam tersi)
    sc = plt.scatter(valid_data[:, 5], valid_data[:, 2], 
                     c=valid_data[:, 0], cmap='jet', 
                     s=10, alpha=0.7)
    
    plt.colorbar(sc, label='Maliyet ($)')
    
    # En iyiyi tekrar işaretle
    plt.scatter(best_design[5], best_design[2], 
                color='white', s=150, marker='*', edgecolors='black', label='En İyi Nokta')

    plt.title('Isı Haritası: Nüve Çapı ve Folyo Yüksekliği İlişkisi')
    plt.xlabel('Nüve Çapı (mm)')
    plt.ylabel('Folyo Yüksekliği (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()

# --- KULLANIM ---
# Hesaplama bittikten hemen sonra bu fonksiyonu çağır:

# --- Kullanım ---
# Önce çizimi oluştur
# create_professional_drawing(best_transformer, "sonuc.dxf")

# Sonra resme çevir
# convert_dxf_to_png("sonuc.dxf", "sonuc_gorunumu.png")
# --- KULLANIM ---
# generate_full_drawing(best_transformer)
if __name__ == '__main__':
    print("Variable Core Length Turbo Search Başlıyor...")
    
    # PARAMETRELER
    # Core Length adımını 5mm veya 10mm tutmak performansı korur. 1mm yaparsan süre uzar.
    CORE_LEN_STEP = 1.0 
    
    core_dias = np.arange(90, 500, 10, dtype=np.float64) 
    turns = np.arange(10, 70, 1, dtype=np.float64)     
    heights = np.arange(200, 1200, 5, dtype=np.float64) 
    thicks = np.arange(0.3, 4.0, 0.05, dtype=np.float64)  
    hvdias = np.arange(1.0, 4.0, 0.05, dtype=np.float64)  


    
    start_time = time.time()
    
    # Fonksiyonu çağır
    all_results = optimize_transformer_variable_length(
        core_dias, turns, heights, thicks, hvdias, 5.0, CORE_LEN_STEP
    )
    
    # En iyiyi bul
    best_idx = np.argmin(all_results[:, 0])
    best_transformer = all_results[best_idx]
    
    end_time = time.time()
    
    print(f"Toplam Süre: {end_time - start_time:.4f} saniye")
    print("-" * 30)
    print(f"EN İYİ FİYAT: {best_transformer[0]:.2f}")
    print(f"Core Diameter: {best_transformer[5]}")
    print(f"Core Length  : {best_transformer[6]} (YENİ)")
    print(f"Turns        : {best_transformer[1]}")
    print(f"Height       : {best_transformer[2]}")
    print(f"Thickness    : {best_transformer[3]}")
    print(f"HV Dia       : {best_transformer[4]}")
    filename = "cizim.dxf"

    visualize_design_space(all_results)
    #create_professional_drawing(best_transformer,filename)
    #pngfilename = "pngcizim.png"
    #convert_dxf_to_png(filename,pngfilename)
