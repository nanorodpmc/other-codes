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


# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.9, 0.025, 0.09, 0.04])
zerobutton = Button(resetax, 'Reset', hovercolor='0.975')
def reset(event):
    a_slider.valinit=init_a ; a_slider.reset()
    b_slider.valinit=init_b ; b_slider.reset()
    c_slider.valinit=init_c ; c_slider.reset()

    theta_slider.valinit=init_theta ; theta_slider.reset()
zerobutton.on_clicked(reset)

pos1ax = plt.axes([0.4, 0.025, 0.09, 0.04])
pos1button = Button(pos1ax, 'pos1', hovercolor='0.975')
def pos1(event):
    a_slider.valinit=0.237 ; a_slider.reset()
    b_slider.valinit=0.090 ; b_slider.reset()
    c_slider.valinit=0.741 ; c_slider.reset()

    theta_slider.valinit=18.01 ; theta_slider.reset()
pos1button.on_clicked(pos1)

pos2ax = plt.axes([0.5, 0.025, 0.09, 0.04])
pos2button = Button(pos2ax, 'pos2', hovercolor='0.975')
def pos2(event):
    a_slider.valinit=0.434 ; a_slider.reset()
    b_slider.valinit=0.180 ; b_slider.reset()
    c_slider.valinit=0.519 ; c_slider.reset()

    theta_slider.valinit=-3.43 ; theta_slider.reset()
pos2button.on_clicked(pos2)

pos3ax = plt.axes([0.6, 0.025, 0.09, 0.04])
pos3button = Button(pos3ax, 'pos3', hovercolor='0.975')
def pos3(event):
    a_slider.valinit=0.230 ; a_slider.reset()
    b_slider.valinit=0.089 ; b_slider.reset()
    c_slider.valinit=0.792 ; c_slider.reset()

    theta_slider.valinit=19.37 ; theta_slider.reset()
pos3button.on_clicked(pos3)

pos4ax = plt.axes([0.7, 0.025, 0.09, 0.04])
pos4button = Button(pos4ax, 'pos4', hovercolor='0.975')
def pos4(event):
    a_slider.valinit=0.259 ; a_slider.reset()
    b_slider.valinit=0.575 ; b_slider.reset()
    c_slider.valinit=0.113 ; c_slider.reset()

    theta_slider.valinit=24.17 ; theta_slider.reset()
pos4button.on_clicked(pos4)

pos5ax = plt.axes([0.8, 0.025, 0.09, 0.04])
pos5button = Button(pos5ax, 'pos5', hovercolor='0.975')
def pos5(event):
    a_slider.valinit=0.034 ; a_slider.reset()
    b_slider.valinit=0.294 ; b_slider.reset()
    c_slider.valinit=0.613 ; c_slider.reset()

    theta_slider.valinit=-9.38 ; theta_slider.reset()
pos5button.on_clicked(pos5)


plt.show()