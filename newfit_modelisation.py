import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# The parametrized function to be plotted
def f(x, a, b,c,theta):
    theta=theta*np.pi/180
    return (a*np.sin(x+theta)**2 + b*np.cos(x+theta)**2 + c*np.sin(2*x+theta)**2) / max([a*np.sin(i+theta)**2 + b*np.cos(i+theta)**2+ c*np.sin(2*i+theta)**2 for i in np.linspace(0,2*np.pi,1000)])

x=np.linspace(0,2*np.pi,1000)


# Define initial parameters
init_a = 0.5
init_b = 0.5
init_c = 0
init_theta=0

# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(projection='polar')


line, = plt.plot(x, f(x, init_a, init_b, init_c, init_theta), lw=2)


# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.4, bottom=0.2)
plt.title('a sin^2 (x) + b cos^2 (x) + c sin^2 (2x)')

axa = plt.axes([0.05, 0.25, 0.0225, 0.63])
a_slider = Slider(
    ax=axa,
    label='a',
    valmin=0,
    valmax=1,
    valinit=init_a,
    orientation="vertical"
)

axb = plt.axes([0.15, 0.25, 0.0225, 0.63])
b_slider = Slider(
    ax=axb,
    label="b",
    valmin=0,
    valmax=1,
    valinit=init_b,
    orientation="vertical"
)

axc = plt.axes([0.25, 0.25, 0.0225, 0.63])
c_slider = Slider(
    ax=axc,
    label="c",
    valmin=-0.2,
    valmax=1,
    valinit=init_c,
    orientation="vertical"
)


axtheta = plt.axes([0.1, 0.05, 0.2, 0.04])
theta_slider = Slider(
    ax=axtheta,
    label="θ",
    valmin=-45,
    valmax=45,
    valinit=init_theta,
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(x, a_slider.val, b_slider.val, c_slider.val, theta_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
a_slider.on_changed(update)
b_slider.on_changed(update)
c_slider.on_changed(update)

theta_slider.on_changed(update)


plt.show()
