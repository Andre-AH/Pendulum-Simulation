
#####################################################################################################
# Importações
from PIL import Image, ImageTk
import customtkinter
from tkinter import *
import plotly.graph_objects as go
import matplotlib.animation as animation
from scipy.integrate import odeint
from scipy.special import ellipk
import matplotlib.pyplot as plt
import numpy as np
import math
from IPython import display

###################################
"Simulação de um pêndulo"
"por André Henriques"
###################################

# Interpretador: Python 3.8 64bits

# Fontes:
# -> https://www.youtube.com/watch?v=0q0L7Fj4dk8
# -> https://www.youtube.com/watch?v=p_di4Zn4wz4&list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6&index=4

#####################################################################################################
# Constantes
velocidade_angular_0 = 0  # Velocidade angular inicial [rad/s]
theta_0 = 0.5  # Ângulo inicial [rad]
L = 1.69  # Comprimento do fio do pêndulo [m]
g = 9.81  # Constante gravitacional [m/s^2]
miu = 0.0  # Coefeciente de perda energética
timestep = 0.001  # Variações de tempo [s]
tempo_total = 10  # Tempo total da simulação [s]

#####################################################################################################
# Modes: system (default), light, dark
customtkinter.set_appearance_mode("dark")
# Themes: blue (default), dark-blue, green
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()

root.title('Simulação de um pêndulo')
root.geometry("350x750")

lista_contador = [0, 0, 0, 0, 0, 0, 0, 0]


def do_something(p):
    p.configure(fg_color="#D35B58")


def contador(n, lista_contador, p):
    if n == 1 and lista_contador[2] == 1:
        lista_contador[n] = 0
    elif n == 2 and lista_contador[1] == 1:
        lista_contador[n] = 0
    else:
        lista_contador[n] += 1
        do_something(p)


# Create Our Buttons
button_1 = customtkinter.CTkButton(master=root, text="Gráfico Ângulo-tempo", width=210,
                                   height=40, compound="top", command=lambda: contador(0, lista_contador, button_1))
button_1.pack(pady=20, padx=20)

button_2 = customtkinter.CTkButton(master=root, text="Animação do Pêndulo (simples)",
                                   width=210, height=40, compound="top", command=lambda: contador(1, lista_contador, button_2))
button_2.pack(pady=20, padx=20)

button_3 = customtkinter.CTkButton(master=root, text="Animação do Pêndulo (completo)",
                                   width=210, height=40, compound="top", command=lambda: contador(2, lista_contador, button_3))
button_3.pack(pady=20, padx=20)

button_4 = customtkinter.CTkButton(master=root, text="Espaço de fase", width=210,
                                   height=40, compound="top", command=lambda: contador(3, lista_contador, button_4))
button_4.pack(pady=20, padx=20)

button_5 = customtkinter.CTkButton(master=root, text="Gráfico interativo - theta inicial",
                                   width=210, height=40, compound="top", command=lambda: contador(4, lista_contador, button_5))
button_5.pack(pady=20, padx=20)

button_6 = customtkinter.CTkButton(master=root, text="Gráfico interativo - gravidade",
                                   width=210, height=40, compound="top", command=lambda: contador(5, lista_contador, button_6))
button_6.pack(pady=20, padx=20)

button_7 = customtkinter.CTkButton(master=root, text="Gráfico interativo - perda energética",
                                   width=210, height=40, compound="top", command=lambda: contador(6, lista_contador, button_7))
button_7.pack(pady=20, padx=20)

button_8 = customtkinter.CTkButton(master=root, text="Gráfico interativo - comprimento",
                                   width=210, height=40, compound="top", command=lambda: contador(7, lista_contador, button_8))
button_8.pack(pady=20, padx=20)

exit_button = customtkinter.CTkButton(
    master=root, text="Seguinte", width=110, height=40, compound="top", command=root.destroy, fg_color="green")
exit_button.pack(pady=20, padx=20)

root.mainloop()

#####################################################################################################

# Equação simples do pêndulo (para pequenos ângulos): dtheta^2/dt^2 = -(g/L) x theta -> theta = theta_0 x cos( ((g/L)^(1/2)) x t)


def equa_simples(theta_0, L, g, t):
    theta = theta_0 * math.cos(((g/L)**(1/2)) * t)
    return theta

#####################################################################################################

# Equação diferencial (completa) do pêndulo: dtheta^2/dt^2 = -(g/L) x sin(theta) - miu x dtheta/dt


def derivadas(x, t):

    return [x[1], -(g/L)*math.sin(x[0]) - miu*x[1]]


def equa_completa(theta_0, L, g, t, miu):
    theta_B = odeint(model, y0, t)
    return theta_B


time = np.arange(0, tempo_total, timestep)
position, velocity = odeint(derivadas, [theta_0, velocidade_angular_0], time).T

#####################################################################################################

# Gráficos -> Theta vs tempo (simples e complexo), velocidade angular vs theta

# Theta vs tempo (simples)

pos = []
for i in time:
    pos.append(equa_simples(theta_0, L, g, i))


def grafico_theta_vs_tempo_1(theta_0, pos, time):
    plt.plot(time, pos, '#DC9D42', linewidth=2)
    plt.xlabel('tempo [s]')
    plt.ylabel('Ângulo [rad]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(True)

# Theta vs tempo (completo)


def grafico_theta_vs_tempo_2(time, position):
    plt.plot(time, position, '#4993D0', linewidth=2)
    plt.xlabel('tempo [s]')
    plt.ylabel('Ângulo [rad]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(True)


plt.style.use('dark_background')
if lista_contador[0] == 1:
    grafico_theta_vs_tempo_1(theta_0, pos, time)
    grafico_theta_vs_tempo_2(time, position)
    plt.title("Ângulos em função do tempo")
    plt.legend(['Simplificada', 'Completa'])
    plt.show()

#####################################################################################################

# Animação do pêndulo (simples e completo)


def get_coords(th):
    """Return the (x, y) coordinates of the bob at angle th."""
    return L * np.sin(th), -L * np.cos(th)


if lista_contador[1] == 1 or lista_contador[2] == 1:
    fig = plt.figure()
    ax = fig.add_subplot(aspect='equal')
    # The pendulum rod, in its initial position.
    x0, y0 = get_coords(theta_0)
    line, = ax.plot([0, x0], [0, y0], lw=3, c='w')
    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08
    circle = ax.add_patch(plt.Circle(get_coords(theta_0), bob_radius,
                                     fc='b', zorder=3))
    # Set the plot limits so that the pendulum has room to swing!
    ax.set_xlim(-L*1.2, L*1.2)
    ax.set_ylim(-L*1.2, L*1.2)
    n = 1


def animate(i):
    """Update the animation at frame i."""
    if n == 0:
        x, y = get_coords(pos[i])
    elif n == 1:
        x, y = get_coords(position[i])
    line.set_data([0, x], [0, y])
    circle.set_center((x, y))


if lista_contador[1] == 1 or lista_contador[2] == 1:
    nframes = int(tempo_total/timestep)
    interval = timestep * 1000
    ani = animation.FuncAnimation(
        fig, animate, frames=nframes, repeat=True, interval=interval)
    plt.grid(True)
if lista_contador[1] == 1:
    plt.title("Ânimação do pêndulo (simples)")
    plt.show()
if lista_contador[2] == 1:
    plt.title("Ânimação do pêndulo (completa)")
    plt.show()
# ani.save('Animação_pêndulo.gif', fps=24)
#####################################################################################################

# Gráfico: Espaço de fase dtheta/dt em função de theta

THETA_0_1 = theta_0
THETA_DOT_0_1 = velocidade_angular_0
P1 = str(input("Quer mudar o ângulo inicial [S/N]: "))
if P1 == "S":
    THETA_0_1 = float(input("Ângulo inicial 1 [rad]: "))
    THETA_0_2 = float(input("Ângulo inicial 2 [rad]: "))
else:
    THETA_0_2 = 0
P2 = str(input("Quer mudar a velocidade angular [S/N]: "))
if P2 == "S":
    THETA_DOT_0_1 = float(input("Velocidade angular inicial 1 [rad/s]: "))
    THETA_DOT_0_2 = float(input("Velocidade angular inicial 2 [rad/s]: "))
else:
    THETA_DOT_0_2 = 0
X_fase_1 = [THETA_0_1]
Y_fase_1 = [THETA_DOT_0_1]
X_fase_2 = [THETA_0_2]
Y_fase_2 = [THETA_DOT_0_2]


def get_theta_double_dot(theta, theta_dot):
    return -miu * theta_dot - (g / L) * np.sin(theta)


def theta(t):
    # theta = [THETA_0_1, THETA_0_2]
    # theta_dot = [THETA_DOT_0_1, THETA_DOT_0_2]
    theta_1 = THETA_0_1
    theta_2 = THETA_0_2
    theta_dot_1 = THETA_DOT_0_1
    theta_dot_2 = THETA_DOT_0_2
    delta_t = 0.001
    for time in np.arange(0, t, delta_t):
        theta_double_dot_1 = get_theta_double_dot(theta_1, theta_dot_1)
        theta_1 += theta_dot_1 * delta_t
        X_fase_1.append(theta_1)
        theta_dot_1 += theta_double_dot_1 * delta_t
        Y_fase_1.append(theta_dot_1)

        theta_double_dot_2 = get_theta_double_dot(theta_2, theta_dot_2)
        theta_2 += theta_dot_2 * delta_t
        X_fase_2.append(theta_2)
        theta_dot_2 += theta_double_dot_2 * delta_t
        Y_fase_2.append(theta_dot_2)
    return X_fase_1, Y_fase_1, X_fase_2, Y_fase_2


def plotdf(f, xran=[-5, 5], yran=[-5, 5], grid=[21, 21], color='k'):
    """
    Plot the direction field for an ODE written in the form 
        x' = F(x,y)
        y' = G(x,y)

    The functions F,G are defined in the list of strings f.

    Input
    -----
    f:    list of strings ["F(X,Y)", "G(X,Y)"
          F,G are functions of X and Y (capitals).
    xran: list [xmin, xmax] (optional)
    yran: list [ymin, ymax] (optional)
    grid: list [npoints_x, npoints_y] (optional)
          Defines the number of points in the x-y grid.
    color: string (optional)
          Color for the vector field (as color defined in matplotlib)
    """
    x = np.linspace(xran[0], xran[1], grid[0])
    y = np.linspace(yran[0], yran[1], grid[1])
    def dX_dt(X, Y, t=0): return map(eval, f)

    X, Y = np.meshgrid(x, y)  # create a grid
    DX, DY = dX_dt(X, Y)        # compute growth rate on the grid
    plt.quiver(X, Y, DX, DY, pivot='mid', color="r")
    plt.xlim(xran), plt.ylim(yran)
    plt.grid('on')
    plt.title("Espaço de fase: campo vetorial do ângulo e velocidade angular")
    plt.xlabel('Ângulo [rad]')
    plt.ylabel('Velocidade angular [rad/s]')

    X_fase_1, Y_fase_1, X_fase_2, Y_fase_2 = theta(30)
    plt.plot(X_fase_1, Y_fase_1, 'w', linewidth=0.8)
    plt.plot(X_fase_2, Y_fase_2, 'y', linewidth=0.8)
    plt.show()


pendulum = ["Y", "(-(g/L)*np.sin(X)) - (miu*Y)"]
if lista_contador[3] == 1:
    plotdf(pendulum, xran=[-4*np.pi, 4*np.pi], yran=[-20, 20])


#####################################################################################################

# Gráficos interativos

# Posição dependendo do ângulo inicial

if lista_contador[4] == 1:
    fig_0 = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, 370, 1):
        position, velocity = odeint(
            derivadas, [(step*3.14)/180, velocidade_angular_0], time).T
        fig_0.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=time,
                y=position))

    # Make 10th trace visible
    fig_0.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig_0.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_0.data)},
                  {"title": "Step: " + str(i) + "-> Ângulo inicial: " + str((i*3.14)/180) + "rad"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=100,
        currentvalue={"prefix": "Ângulo: "},
        pad={"t": 50},
        steps=steps
    )]

    fig_0.update_layout(
        sliders=sliders
    )

    fig_0.show()


# Posição dependendo da gravidade

if lista_contador[5] == 1:
    fig_1 = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, 100, 1):
        g = step
        position, velocity = odeint(
            derivadas, [theta_0, velocidade_angular_0], time).T
        fig_1.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=time,
                y=position))

    # Make 10th trace visible
    fig_1.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig_1.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_1.data)},
                  {"title": "Step: " + str(i) + "-> Aceleração gravítica:" + str(i) + "m/s^2"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=100,
        currentvalue={"prefix": "g: "},
        pad={"t": 50},
        steps=steps
    )]

    fig_1.update_layout(
        sliders=sliders
    )

    fig_1.show()


# Posição dependendo do coefeciente de perda energética

if lista_contador[6] == 1:
    fig_2 = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, 100, 1):
        miu = step/100
        position, velocity = odeint(
            derivadas, [theta_0, velocidade_angular_0], time).T
        fig_2.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=time,
                y=position))

    # Make 10th trace visible
    fig_2.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig_2.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_2.data)},
                  {"title": "Step: " + str(i) + "-> Coefeciente de perda energética:" + str(i/100)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=100,
        currentvalue={"prefix": "miu: "},
        pad={"t": 50},
        steps=steps
    )]

    fig_2.update_layout(
        sliders=sliders
    )

    fig_2.show()


# Posição dependendo do comprimento do fio

if lista_contador[7] == 1:
    fig_2 = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(1, 11, 1):
        miu = 0
        L = step
        position, velocity = odeint(
            derivadas, [theta_0, velocidade_angular_0], time).T
        fig_2.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=time,
                y=position))

    # Make 10th trace visible
    fig_2.data[1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig_2.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig_2.data)},
                  {"title": "Step: " + str(i) + "-> Comprimento do fio:" + str(i) + "m"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=100,
        currentvalue={"prefix": "L: "},
        pad={"t": 50},
        steps=steps
    )]

    fig_2.update_layout(
        sliders=sliders
    )

    fig_2.show()


#####################################################################################################
