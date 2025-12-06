import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simulation import simulate_sveir  # <-- corrected import

st.set_page_config(page_title='SVEIR Worm Simulator', layout='wide')
st.title('SVEIR Stochastic Worm Simulator')

# UI: sliders
beta = st.slider('Infection rate (beta)', 0.0, 2.0, 0.3, 0.01)
nu = st.slider('Vaccination/patching rate (nu)', 0.0, 1.0, 0.05, 0.01)
kappa = st.slider('Incubation rate (kappa)', 0.0, 1.0, 0.2, 0.01)
gamma = st.slider('Recovery rate (gamma)', 0.0, 1.0, 0.1, 0.01)
sigma = st.slider('Noise intensity (sigma)', 0.0, 1.0, 0.05, 0.01)
N = st.number_input('Total nodes N', min_value=10, value=1000, step=10)

S0 = st.number_input('Initial susceptible S0', min_value=0, value=int(N-20), step=1)
I0 = st.number_input('Initial infected I0', min_value=0, value=10, step=1)
E0 = st.number_input('Initial exposed E0', min_value=0, value=5, step=1)

T = st.number_input('Simulation time T', min_value=1, value=100)
dt = st.number_input('Time step dt', min_value=0.001, value=0.1)

if st.button('Run simulation'):

    params = dict(beta=beta, nu=nu, kappa=kappa, gamma=gamma, sigma=sigma, N=N,
                  S0=S0, I0=I0, E0=E0, V0=0, R0=0)

    df = simulate_sveir(params, T=T, dt=dt)

    st.line_chart(df.set_index('t')[['S','E','I','R','V']])

    final_I = df['I'].iloc[-1]
    st.write(f'Final infected: {final_I:.3f}')

    if final_I < 1:
        st.success('Worm likely extinct (I < 1 at final time)')
    else:
        st.warning('Worm persists at final time')

    csv = df.to_csv(index=False)
    st.download_button('Download simulation CSV', data=csv, file_name='simulation.csv')
