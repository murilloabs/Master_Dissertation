# %%
# ==============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES DE DIRETÓRIOS
# ==============================================================================
import sys
sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\ross')
import pickle
import os
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import coherence
import ross as rs

# Ajuste os caminhos conforme sua máquina
# sys.path.append(r'C:\Users\Murillo\OneDrive - Universidade Federal de Uberlândia\Área de Trabalho\Mestrado\ENGRENAMENTO\Implementacao\ross')
# sys.path.append(r'D:\MESTRADO\Backlash_Devlopment-main\Backlash_Devlopment-main\codes')

from backlash import Backlash, compute_dfft

CSV_DIR = r"C:\\Users\\Murillo\\OneDrive - Universidade Federal de Uberlândia\\Área de Trabalho\\Mestrado\\ENGRENAMENTO\\Implementacao\\teste_backlash_gemini\\validation\\paper"


try:
    main_file = sys.modules['__main__'].__file__
    diretorio_execucao = os.path.dirname(os.path.abspath(main_file))
except AttributeError:
    diretorio_execucao = os.getcwd()

OUTPUT_PLOT_DIR = os.path.join(diretorio_execucao, "validation_plots")
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

print(f"Diretório de saída: {OUTPUT_PLOT_DIR}")

METRICS_FILE = os.path.join(OUTPUT_PLOT_DIR, "validation_metrics_report.txt")
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    f.write("=========================================================\n")
    f.write("RELATÓRIO DE MÉTRICAS DE VALIDAÇÃO ESTATÍSTICA\n")
    f.write("=========================================================\n\n")

# === NOVO: ARMAZENAMENTO GLOBAL PARA A TABELA LATEX ===
GLOBAL_METRICS = []

COLOR_SIM = '#0072B2'   
COLOR_PAPER = '#D55E00' 
COLOR_REF = '#000000'   

# %%
# ==============================================================================
# FUNÇÕES AUXILIARES PARA VALIDAÇÃO E PLOTAGENS COMUNS
# ==============================================================================

def load_paper_data(filename):
    filepath = os.path.join(CSV_DIR, filename)
    if not os.path.exists(filepath):
        print(f"AVISO: Arquivo do paper não encontrado: {filepath}. Plotará apenas a simulação.")
        return None, None
    try:
        df = pd.read_csv(filepath, header=None)
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except Exception as e:
        print(f"Erro ao ler {filename}: {e}")
        return None, None

def save_plot_formats(fig, base_filename):
    html_path = os.path.join(OUTPUT_PLOT_DIR, base_filename)
    pdf_path = os.path.join(OUTPUT_PLOT_DIR, base_filename.replace('.html', '.pdf'))
    fig.write_html(html_path)
    try:
        fig.write_image(pdf_path, format='pdf')
    except Exception:
        pass 

def plot_validation(sim_x, sim_y, csv_filename, title, x_label, y_label, output_filename, 
                    x_range=None, sim_scale_y=1.0, paper_scale_x=1.0, paper_scale_y=1.0, 
                    step_sim=1, step_paper=1, constant_ref=None,
                    rpm=None, var_name=None):  # <-- NOVO: Parâmetros para identificar a tabela
    
    fig = go.Figure()
    sim_y_scaled = sim_y * sim_scale_y

    if x_range:
        mask = (sim_x >= x_range[0]) & (sim_x <= x_range[1])
        x_plot = sim_x[mask][::step_sim]
        y_plot = sim_y_scaled[mask][::step_sim]
    else:
        x_plot = sim_x[::step_sim]
        y_plot = sim_y_scaled[::step_sim]

    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='Modelo Implementado', 
                             line=dict(color=COLOR_SIM, width=2.5)))

    p_x, p_y = load_paper_data(csv_filename)
    if p_x is not None and p_y is not None:
        p_x_scaled = (p_x * paper_scale_x)[::step_paper]
        p_y_scaled = (p_y * paper_scale_y)[::step_paper]
        
        fig.add_trace(go.Scatter(x=p_x_scaled, y=p_y_scaled, 
                                 mode='markers', name='Yi et al. (2019)', 
                                 marker=dict(color=COLOR_PAPER, size=5, symbol='circle', opacity=0.85)))

        y_sim_interpolado = np.interp(p_x_scaled, sim_x, sim_y_scaled)
        
        if x_range:
            mask_calc = (p_x_scaled >= x_range[0]) & (p_x_scaled <= x_range[1])
            y_real = p_y_scaled[mask_calc]
            y_pred = y_sim_interpolado[mask_calc]
        else:
            y_real = p_y_scaled
            y_pred = y_sim_interpolado

        if len(y_real) > 0:
            erro = y_pred - y_real
            mae = np.mean(np.abs(erro))
            rmse = np.sqrt(np.mean(erro**2))
            amplitude = np.max(y_real) - np.min(y_real)
            nrmse = (rmse / amplitude) * 100 if amplitude != 0 else 0
            max_err = np.max(np.abs(erro))

            # Salva no TXT geral
            with open(METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{title}]\n")
                f.write(f" - MAE         : {mae:.4f}\n")
                f.write(f" - RMSE        : {rmse:.4f}\n")
                f.write(f" - NRMSE       : {nrmse:.2f} %\n")
                f.write(f" - Erro Máximo : {max_err:.4f}\n\n")

            # === NOVO: Salva os dados na memória para gerar o LaTeX (1000 e 3000 RPM) ===
            if rpm in [1000, 3000] and var_name:
                GLOBAL_METRICS.append({
                    'var': var_name,
                    'rpm': rpm,
                    'mae': mae,
                    'rmse': rmse,
                    'nrmse': nrmse
                })

    if constant_ref is not None:
        fig.add_trace(go.Scatter(x=[x_plot[0], x_plot[-1]], y=[constant_ref, constant_ref], 
                                 mode='lines', name='Valor de referência (Constante)', 
                                 line=dict(color=COLOR_REF, width=2, dash='dash')))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white",
                      legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    if x_range: fig.update_xaxes(range=x_range)

    save_plot_formats(fig, output_filename)


def plot_fft_validation(sim_time, sim_signal, csv_filename, title, x_label, y_label, output_filename, 
                        sim_scale_y=1.0, x_range=None, paper_scale_x=1.0, paper_scale_y=1.0, step_paper=1):
    fig = go.Figure()
    meio = len(sim_time) // 2
    freq_hz, amp = compute_dfft(sim_signal[meio:], sim_time[meio:], freq_unit="Hz", window="hann")
    freq_khz = freq_hz / 1000.0  
    amp_scaled = amp * sim_scale_y

    fig.add_trace(go.Scatter(x=freq_khz, y=amp_scaled, mode='lines', name='Modelo Implementado (FFT)', line=dict(color=COLOR_SIM, width=2)))

    p_x, p_y = load_paper_data(csv_filename)
    if p_x is not None and p_y is not None:
        p_x_scaled = (p_x * paper_scale_x)[::step_paper]
        p_y_scaled = (p_y * paper_scale_y)[::step_paper]
        fig.add_trace(go.Scatter(x=p_x_scaled, y=p_y_scaled, mode='markers', name='Yi et al. (2019)', marker=dict(color=COLOR_PAPER, size=4, opacity=0.85)))

        y_sim_interpolado = np.interp(p_x_scaled, freq_khz, amp_scaled)
        if x_range:
            mask_calc = (p_x_scaled >= x_range[0]) & (p_x_scaled <= x_range[1])
            y_real = p_y_scaled[mask_calc]
            y_pred = y_sim_interpolado[mask_calc]
        else:
            y_real = p_y_scaled
            y_pred = y_sim_interpolado

        if len(y_real) > 0:
            erro = y_pred - y_real
            mae = np.mean(np.abs(erro))
            rmse = np.sqrt(np.mean(erro**2))
            amplitude = np.max(y_real) - np.min(y_real)
            nrmse = (rmse / amplitude) * 100 if amplitude != 0 else 0
            max_err = np.max(np.abs(erro))

            with open(METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{title}]\n - MAE         : {mae:.4f}\n - RMSE        : {rmse:.4f}\n - NRMSE       : {nrmse:.2f} %\n - Erro Máximo : {max_err:.4f}\n\n")

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    if x_range: fig.update_xaxes(range=x_range)
    save_plot_formats(fig, output_filename)


def plot_coherence_validation(sim_time, sim_signal, csv_filename, title, output_filename, sim_scale_y=1.0, paper_scale_x=1.0, paper_scale_y=1.0, step_paper=1):
    fig = go.Figure()
    sim_y_scaled = sim_signal * sim_scale_y
    p_x, p_y = load_paper_data(csv_filename)
    if p_x is not None and p_y is not None:
        p_x_scaled = (p_x * paper_scale_x)[::step_paper]
        p_y_scaled = (p_y * paper_scale_y)[::step_paper]
        y_sim_interpolado = np.interp(p_x_scaled, sim_time, sim_y_scaled)
        dt_mean = np.mean(np.diff(p_x_scaled))
        if dt_mean > 0:
            fs = 1.0 / dt_mean
            n_points = len(p_x_scaled)
            nperseg_val = min(256, max(n_points // 4, 4)) 
            f, Cxy = coherence(p_y_scaled, y_sim_interpolado, fs=fs, nperseg=nperseg_val)
            fig.add_trace(go.Scatter(x=f / 1000.0, y=Cxy, mode='lines', name='Coerência (Modelo x Paper)', line=dict(color=COLOR_SIM, width=2.5)))
            fig.update_layout(title=title, xaxis_title="Frequência f /kHz", yaxis_title="Coerência", template="plotly_white", yaxis=dict(range=[-0.05, 1.1]))
            save_plot_formats(fig, output_filename)
        else:
            print(f"AVISO: Tempo inconsistente no {csv_filename}, não foi possível calcular Coerência.")

def plot_poincare_validation(pasta_saida_simulacao, csv_filename_paper, title, x_label, y_label, output_filename, paper_scale_x=1.0, paper_scale_y=1.0):
    fig = go.Figure()
    poincare_csv = os.path.join(pasta_saida_simulacao, "nao_linear_dados_poincare.csv")
    if os.path.exists(poincare_csv):
        df_sim = pd.read_csv(poincare_csv)
        p_delta_um = df_sim["Delta_um"].values
        p_delta_dot_mms = df_sim["dDelta_dt_mm_s"].values
        fig.add_trace(go.Scatter(x=p_delta_um, y=p_delta_dot_mms, mode='markers', name='Modelo Implementado (Poincaré)', marker=dict(color=COLOR_SIM, size=8, symbol='circle-open', line_width=2, opacity=0.9)))
    else:
        print(f"AVISO: Arquivo nativo do Poincaré não encontrado em {poincare_csv}")
    p_x, p_y = load_paper_data(csv_filename_paper)
    if p_x is not None and p_y is not None:
        fig.add_trace(go.Scatter(x=p_x * paper_scale_x, y=p_y * paper_scale_y, mode='markers', name='Yi et al. (2019)', marker=dict(color=COLOR_PAPER, size=7, symbol='x', opacity=0.9)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor="black", borderwidth=1))
    save_plot_formats(fig, output_filename)

# === NOVO: FUNÇÃO PARA GERAR O CÓDIGO DA TABELA LATEX ===
def generate_latex_table(metrics_list, output_dir):
    """
    Agrupa as métricas salvas e gera o código de uma tabela LaTeX 
    formatada e pronta para o artigo/dissertação.
    """
    if not metrics_list:
        return
        
    latex_str = r"""\begin{table}[H]
	\centering
	\begin{threeparttable}
		\caption{Métricas estatísticas de validação do modelo analítico (Domínio do Tempo)}
		\label{tab:metricas_validacao}
		
		\begin{tabular}{llccc}
			\hline
			Variável & Rotação & MAE & RMSE & NRMSE (\%) \\
			\hline
"""
    
    # Identificar quais variáveis únicas foram registradas
    variaveis_unicas = []
    for m in metrics_list:
        if m['var'] not in variaveis_unicas:
            variaveis_unicas.append(m['var'])
            
    for var in variaveis_unicas:
        # Puxar dados de 1000 e 3000 RPM para a variável atual
        m1000 = next((m for m in metrics_list if m['var'] == var and m['rpm'] == 1000), None)
        m3000 = next((m for m in metrics_list if m['var'] == var and m['rpm'] == 3000), None)
        
        # Monta a estrutura elegante mantendo a primeira coluna vazia na segunda linha
        if m1000 and m3000:
            latex_str += f"\t\t\t{var} & 1000 rpm & ${m1000['mae']:.4f}$ & ${m1000['rmse']:.4f}$ & ${m1000['nrmse']:.2f}$ \\\\\n"
            latex_str += f"\t\t\t & 3000 rpm & ${m3000['mae']:.4f}$ & ${m3000['rmse']:.4f}$ & ${m3000['nrmse']:.2f}$ \\\\\n"
            latex_str += "\t\t\t\\hline\n"
            
    latex_str += r"""		\end{tabular}
		
		\begin{tablenotes}[para, flushleft]
			\small
			\item Fonte: Elaborado pelo autor.
		\end{tablenotes}
		
	\end{threeparttable}
\end{table}
"""
    
    table_path = os.path.join(output_dir, "tabela_latex_metricas.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"\n[OK] Código da Tabela LaTeX exportado com sucesso em: {table_path}")


# %%
# ==============================================================================
# FUNÇÃO MESTRE PARA CONSTRUIR O MODELO E RODAR A SIMULAÇÃO POR ROTAÇÃO
# ==============================================================================

def run_simulation_at_speed(speed_rpm, sim_time_seconds, integrador="internal_newmark"):
    print(f"\n[{speed_rpm} RPM] Construindo o modelo e calculando n_cicles...")
    
    z1 = z2 = 20
    m_n = 0.01                           
    pd_gear = m_n * z1                   
    alpha_0_rad = np.radians(20.0)
    width = 0.030
    b0 = 50e-6                           
    err_amp = 20e-6                      
    m_gear = 6.57                        
    J_gear = 0.0365                      
    k_brg = 1.0e8                        
    c_brg = 512.64

    ks = 3.6228e8                        
    kd = 6.5072e8                                              

    T10, T1a = 300.0, 100.0
    T20, T2a = 300.0, 100.0

    speed_rad_s = speed_rpm * np.pi / 30
    omega_m = speed_rad_s           
    Tm = 2 * np.pi / (omega_m)      

    n_cicles = int(np.ceil(sim_time_seconds / Tm))
    cut_cicles = 0 
    
    steel = rs.Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)
    steel_stiff = rs.Material(name="Steel_Stiff", rho=0.01, E=1e15, Poisson=0.3)
    
    shaft1 = [rs.ShaftElement(L=0.0001, idl=0.0, odl=0.0001, material=steel_stiff, n=0)]
    brg1 = rs.BearingElement(n=0, kxx=k_brg, kyy=k_brg, cxx=c_brg, cyy=c_brg)
    
    gear1 = rs.GearElementTVMS(
        n=0, material=steel, width=width, bore_diameter=np.sqrt(pd_gear**2-(4*m_gear)/(np.pi*width*steel.rho)), 
        module=m_n, n_teeth=z1, pr_angle=alpha_0_rad, helix_angle=0,
        addendum_coeff=1, tip_clearance_coeff=0.25
    )

    gear1.m = m_gear
    gear1.Ip = J_gear
    gear1.Id = 0.0001*J_gear / 2

    rotor1 = rs.Rotor(shaft_elements=shaft1, disk_elements=[gear1], bearing_elements=[brg1])
    rotor2 = copy.deepcopy(rotor1)

    multirotor = rs.MultiRotor(
        driving_rotor=rotor1, driven_rotor=rotor2, coupled_nodes=(0,0),
        update_mesh_stiffness=True, square_varying_stiffness=True,
        square_stiffness_amplitude_ratio=0.275, orientation_angle=0.0, position="above"
    )

    unb_node = [int(e.n) for e in multirotor.disk_elements if isinstance(e, rs.GearElement)]
    unb_magnitude = [0.0, 0.0]
    unb_phase = [0.0, 0.0]

    backlash = Backlash(
        multirotor, speed_rad_s, b0=b0, error_amp=err_amp, gear_mesh_stiffness=None,
        num_points_cicle=6000, n_cicles=n_cicles, cut_cicles=cut_cicles,
        use_multirotor_coupling_stiffness=False, compute_contact_ratio=True, mesh_damping_ratio=0.07
    )

    _, _, _ = backlash._get_or_create_stiffness_table(square_varying_stiffness=True, kd=kd, ks=ks, n_poits = 1000)

    w1 = speed_rad_s
    w2 = multirotor.mesh.gear_ratio * w1
    F = np.zeros((len(backlash.time), multirotor.ndof))
    F[:, unb_node[0] * multirotor.number_dof + 5] = T10 + T1a * np.sin(w1 * backlash.time)
    F[:, unb_node[1] * multirotor.number_dof + 5] = T20 + T2a * np.sin(w2 * backlash.time)

    print(f"[{speed_rpm} RPM] Iniciando Integração Não-Linear ({n_cicles} ciclos até {sim_time_seconds}s)...")

    gamma = 0.5
    beta = (1/4) * (gamma + 0.5)**2
    
    backlash.run_dynamic_backlash(
        unb_node=unb_node, unb_magnitude=unb_magnitude, unb_phase=unb_phase,
        integration_method=integrador, gamma=gamma, beta=beta, tol=1e-6,
        sigma=1e5, smooth_operator=True, add_force=F
    )
    
    pasta_base_nativa = os.path.join(diretorio_execucao, f"resultados_engrenamento_{speed_rpm}rpm")
    pasta_saida = backlash.save_results(unb_node, unb_magnitude, unb_phase, integrador, output_dir=pasta_base_nativa)
    
    caminho_dash = os.path.join(pasta_saida, "painel_grafico.html")
    backlash.plot_dashboard(
        freq_unit="rpm", decimation=5, save_path=caminho_dash, 
        dft_y_scale="linear", time_range=(2.0, 2.125), freq_range=(0, 400000)
    )

    wm = speed_rad_s * z1
    Tm1 = (2.0 * np.pi) / wm  
    t_max = backlash.time[-1]
    n_periods = int(t_max / Tm1)
    discard_periods = int(n_periods * 0.975)

    backlash.plot_poincare_map(is_linear=False, save_dir=pasta_saida, discard_periods=discard_periods)
    
    idx_x1 = unb_node[0] * multirotor.number_dof + 0
    
    return backlash, idx_x1, pd_gear, b0, alpha_0_rad, multirotor.mesh.contact_ratio, pasta_saida


# %%
# ==============================================================================
# VALIDAÇÃO FIGURAS 7 e 9: ROTAÇÃO BAIXA (1000 RPM)
# ==============================================================================
bk_1000, idx_x1_1000, pd_gear, b0, alfa0_rad, cr0, pasta_saida_1000 = run_simulation_at_speed(speed_rpm=1000, sim_time_seconds=2.65)

t_1000 = bk_1000.time
x1_1000 = bk_1000.time_response.yout[:, idx_x1_1000]
delta_1000 = bk_1000.backlash_results['delta']
alfa_1000_deg = np.degrees(bk_1000.backlash_results['alfa'])

step_sim = 1
step_paper = 1

d0_ref_mm = pd_gear * 1000.0                 
alfa0_ref_deg = np.degrees(alfa0_rad)        
b0_ref_um = b0 * 1e6                         
cr0_ref = cr0                                

# Observe a passagem dos parâmetros 'rpm' e 'var_name' apenas onde queremos extrair para a tabela Latex
plot_validation(t_1000, bk_1000.backlash_results['d'], "7a.csv", "Fig 7(a) - Distância entre centros a 1000 RPM", "Tempo t /s", "Distância d /mm", "Fig_7a_CenterDistance.html", [2.4, 2.6], 1000.0, step_sim=step_sim, step_paper=step_paper, constant_ref=d0_ref_mm)
plot_validation(t_1000, alfa_1000_deg, "7b.csv", "Fig 7(b) - Ângulo de pressão a 1000 RPM", "Tempo t /s", "Ângulo α /(°)", "Fig_7b_PressureAngle.html", [2.4, 2.6], 1.0, step_sim=step_sim, step_paper=step_paper, constant_ref=alfa0_ref_deg)
plot_validation(t_1000, bk_1000.backlash_results['bt'], "7c.csv", "Fig 7(c) - Folga a 1000 RPM", "Tempo t /s", "Folga bt /μm", "Fig_7c_Backlash.html", [2.4, 2.6], 1e6, step_sim=step_sim, step_paper=step_paper, constant_ref=b0_ref_um)
plot_validation(t_1000, bk_1000.backlash_results['contact_ratio'], "7d.csv", "Fig 7(d) - Razão de contato a 1000 RPM", "Tempo t /s", "Razão mp", "Fig_7d_ContactRatio.html", [2.4, 2.6], 1.0, step_sim=step_sim, step_paper=step_paper, constant_ref=cr0_ref)

plot_validation(t_1000, x1_1000, "9a.csv", "Fig 9(a) - Resposta x1 a 1000 RPM", "Tempo t /s", "Deslocamento x1 /μm", "Fig_9a_Vibration_x1.html", [2.4, 2.52], 1e6, step_sim=step_sim, step_paper=step_paper, rpm=1000, var_name="$x_1$ ($\mu$m)")
plot_validation(t_1000, delta_1000, "9c.csv", "Fig 9(c) - DTE a 1000 RPM", "Tempo t /s", "DTE δ /μm", "Fig_9c_Vibration_DTE.html", [2.4, 2.52], 1e6, step_sim=step_sim, step_paper=step_paper, rpm=1000, var_name="$\delta$ ($\mu$m)")
plot_validation(t_1000, bk_1000.backlash_results['Fm'], "9f.csv", "Fig 9(f) - DMF a 1000 RPM", "Tempo t /s", "DMF /kN", "Fig_9f_DMF.html", [2.4, 2.52], 1e-3, step_sim=step_sim, step_paper=step_paper, rpm=1000, var_name="DMF (kN)")

plot_fft_validation(t_1000, x1_1000, "9b.csv", "Espectro FFT x1 a 1000 RPM", "Frequência f /kHz", "Amplitude x1 /μm", "Fig_9b_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 0.8], step_paper=step_paper)
plot_fft_validation(t_1000, delta_1000, "9d.csv", "Espectro FFT δ a 1000 RPM", "Frequência f /kHz", "Amplitude δ /μm", "Fig_9d_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 6], step_paper=step_paper)

plot_poincare_validation(pasta_saida_1000, "9e.csv", "Poincaré δ a 1000 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_9e_Poincare.html")
plot_coherence_validation(t_1000, x1_1000, "9a.csv", "Coerência x1 (1000 RPM)", "Fig_9_Coherence_x1.html", sim_scale_y=1e6)
plot_coherence_validation(t_1000, delta_1000, "9c.csv", "Coerência δ (1000 RPM)", "Fig_9_Coherence_DTE.html", sim_scale_y=1e6)


# %%
# ==============================================================================
# VALIDAÇÃO FIGURA 10: ROTAÇÃO MÉDIA (3000 RPM)
# ==============================================================================
bk_3000, idx_x1_3000, pd_gear, b0, alfa0_rad, cr0, pasta_saida_3000 = run_simulation_at_speed(speed_rpm=3000, sim_time_seconds=2.65)

t_3000 = bk_3000.time
x1_3000 = bk_3000.time_response.yout[:, idx_x1_3000]
delta_3000 = bk_3000.backlash_results['delta']

plot_validation(t_3000, x1_3000, "10a.csv", "Fig 10(a) - Resposta x1 a 3000 RPM", "Tempo t /s", "Deslocamento x1 /μm", "Fig_10a_Vibration_x1.html", [0.8, 0.85], 1e6, step_sim=step_sim, step_paper=step_paper, rpm=3000, var_name="$x_1$ ($\mu$m)")
plot_validation(t_3000, delta_3000, "10c.csv", "Fig 10(c) - DTE a 3000 RPM", "Tempo t /s", "DTE δ /μm", "Fig_10c_Vibration_DTE.html", [0.8, 0.85], 1e6, step_sim=step_sim, step_paper=step_paper, rpm=3000, var_name="$\delta$ ($\mu$m)")
plot_validation(t_3000, bk_3000.backlash_results['Fm'], "10f.csv", "Fig 10(f) - DMF a 3000 RPM", "Tempo t /s", "DMF /kN", "Fig_10f_DMF.html", [0.8, 0.85], 1e-3, step_sim=step_sim, step_paper=step_paper, rpm=3000, var_name="DMF (kN)")

plot_fft_validation(t_3000, x1_3000, "10b.csv", "Espectro FFT x1 a 3000 RPM", "Frequência f /kHz", "Amplitude x1 /μm", "Fig_10b_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 4], step_paper=step_paper)
plot_fft_validation(t_3000, delta_3000, "10d.csv", "Espectro FFT δ a 3000 RPM", "Frequência f /kHz", "Amplitude δ /μm", "Fig_10d_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 7], step_paper=step_paper)

plot_poincare_validation(pasta_saida_3000, "10e.csv", "Poincaré δ a 3000 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_10e_Poincare.html")
plot_coherence_validation(t_3000, x1_3000, "10a.csv", "Coerência x1 (3000 RPM)", "Fig_10_Coherence_x1.html", sim_scale_y=1e6)
plot_coherence_validation(t_3000, delta_3000, "10c.csv", "Coerência δ (3000 RPM)", "Fig_10_Coherence_DTE.html", sim_scale_y=1e6)


# %%
# ==============================================================================
# VALIDAÇÃO FIGURA 11: ROTAÇÃO ALTA (4500 RPM)
# ==============================================================================
bk_4500, idx_x1_4500, pd_gear, b0, alfa0_rad, cr0, pasta_saida_4500 = run_simulation_at_speed(speed_rpm=4500, sim_time_seconds=2.65)

t_4500 = bk_4500.time
x1_4500 = bk_4500.time_response.yout[:, idx_x1_4500]
delta_4500 = bk_4500.backlash_results['delta']

plot_validation(t_4500, x1_4500, "11a.csv", "Fig 11(a) - Resposta x1 a 4500 RPM", "Tempo t /s", "Deslocamento x1 /μm", "Fig_11a_Vibration_x1.html", [0.55, 0.58], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_4500, delta_4500, "11c.csv", "Fig 11(c) - DTE a 4500 RPM", "Tempo t /s", "DTE δ /μm", "Fig_11c_Vibration_DTE.html", [0.55, 0.58], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_4500, bk_4500.backlash_results['Fm'], "11f.csv", "Fig 11(f) - DMF a 4500 RPM", "Tempo t /s", "DMF /kN", "Fig_11f_DMF.html", [0.55, 0.58], 1e-3, step_sim=step_sim, step_paper=step_paper)

plot_fft_validation(t_4500, x1_4500, "11b.csv", "Espectro FFT x1 a 4500 RPM", "Frequência f /kHz", "Amplitude x1 /μm", "Fig_11b_FFT_x1.html", sim_scale_y=1e6, x_range=[0, 3], step_paper=step_paper)
plot_fft_validation(t_4500, delta_4500, "11d.csv", "Espectro FFT δ a 4500 RPM", "Frequência f /kHz", "Amplitude δ /μm", "Fig_11d_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 7], step_paper=step_paper)

plot_poincare_validation(pasta_saida_4500, "11e.csv", "Poincaré δ a 4500 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_11e_Poincare.html")
plot_coherence_validation(t_4500, x1_4500, "11a.csv", "Coerência x1 (4500 RPM)", "Fig_11_Coherence_x1.html", sim_scale_y=1e6)
plot_coherence_validation(t_4500, delta_4500, "11c.csv", "Coerência δ (4500 RPM)", "Fig_11_Coherence_DTE.html", sim_scale_y=1e6)


# %%
# ==============================================================================
# VALIDAÇÃO FIGURA 12: ROTAÇÃO MUITO ALTA (6000 RPM)
# ==============================================================================
bk_6000, idx_x1_6000, pd_gear_6000, b0_6000, alfa0_rad_6000, cr0_6000, pasta_saida_6000 = run_simulation_at_speed(speed_rpm=6000, sim_time_seconds=2.65)

t_6000 = bk_6000.time
delta_6000 = bk_6000.backlash_results['delta']

plot_validation(t_6000, delta_6000, "12a.csv", "Fig 12(a) - DTE a 6000 RPM", "Tempo t /s", "DTE δ /μm", "Fig_12a_Vibration_DTE.html", [0.4, 0.425], 1e6, step_sim=step_sim, step_paper=step_paper)
plot_validation(t_6000, bk_6000.backlash_results['Fm'], "12d.csv", "Fig 12(d) - DMF a 6000 RPM", "Tempo t /s", "DMF /kN", "Fig_12d_DMF.html", [0.4, 0.425], 1e-3, step_sim=step_sim, step_paper=step_paper)

plot_fft_validation(t_6000, delta_6000, "12b.csv", "Espectro FFT δ a 6000 RPM", "Frequência f /kHz", "Amplitude δ /μm", "Fig_12b_FFT_DTE.html", sim_scale_y=1e6, x_range=[0, 8], step_paper=step_paper)
plot_poincare_validation(pasta_saida_6000, "12c.csv", "Poincaré δ a 6000 RPM", "δ /μm", "dδ/dt /(mm/s)", "Fig_12c_Poincare.html")
plot_coherence_validation(t_6000, delta_6000, "12a.csv", "Coerência δ (6000 RPM)", "Fig_12_Coherence_DTE.html", sim_scale_y=1e6)

# ==============================================================================
# GERAÇÃO DA TABELA LATEX FINAL
# ==============================================================================
generate_latex_table(GLOBAL_METRICS, OUTPUT_PLOT_DIR)