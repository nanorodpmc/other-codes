import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter, wiener
from sklearn.decomposition import PCA as sk_pca
from sklearn.decomposition import KernelPCA as sk_kerpca
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Import data from csv
data = pd.read_csv('C:/Users/morel/OneDrive/Bureau/data/20220222/nem monazite EG/10x NA 0.3/images/Spectrum/image_Pos0_Polarizer0_X-Axis0000_Y-Axis0000.csv', sep=',')

wav = [float(a) for a in list(data)[1:-1]]
val = list(list(data.values[:][0])[1:-1])
feat = np.array(val)

objectives = ['10x NA 0.3', '20x NA 0.75',
              '40x NA 0.6', '60x oil NA 1.4', '100x NA 0.9']
mines = [1000]*5
maxes = [0]*5

print('finding min/max to normalize')
for i in range(0, 37):
    for j in objectives:
        data = pd.read_csv('C:/Users/morel/OneDrive/Bureau/data/20220222/nem monazite EG/'+j +
                           '/images/Spectrum/image_Pos0_Polarizer'+str(i)+'_X-Axis0000_Y-Axis0000.csv', sep=',')  # Import data from csv
        val = list(data.values[:][0])[1:-1]
        mines[objectives.index(j)] = min(mines[objectives.index(j)], min(val))
        maxes[objectives.index(j)] = max(maxes[objectives.index(j)], max(val))

print(mines)
print(maxes)
print('reading files')
for i in range(0, 37):
    for j in objectives:
        data = pd.read_csv('C:/Users/morel/OneDrive/Bureau/data/20220222/nem monazite EG/'+j +
                           '/images/Spectrum/image_Pos0_Polarizer'+str(i)+'_X-Axis0000_Y-Axis0000.csv', sep=',')  # Import data from csv
        wav = [float(a) for a in list(data)[1:-1]]
        val = list(data.values[:][0])[1:-1]
        wav.reverse()
        val.reverse()
        val = [a-mines[objectives.index(j)] for a in val]
        # val=[a/(maxes[objectives.index(j)]-mines[objectives.index(j)]) for a in val]
        val = wiener(val, 10)
        feat = np.vstack([feat, val])

feat = feat[1:]


numb = [65, 126, 160, 386, 430, 500, 1356, 1526, 900]

numbers = [a for a in range(1600)]

df = pd.DataFrame()
for w in range(0, 37):
    name = 'angle'+str(w)
    new = pd.DataFrame()
    liste = []
    for i in numbers:
        for r in range(5):
            liste.append(feat[w*5+r][i])
    new[name] = liste
    df = pd.concat([df, new[[name]]], axis=1)

names = []
for a in numbers:
    for i in range(5):
        names.append(wav[a])
an = pd.DataFrame()
an['target'] = names

df = pd.concat([df, an[['target']]], axis=1)

print('in data frame')

features = ['angle'+str(w) for w in range(37)]
x = df.loc[:, features].values  # Separating out the target
y = df.loc[:, ['target']].values  # Standardizing the features

for i in range(len(x)):
    m = max(x[i])
    x[i] = [l/m for l in x[i]]


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC 1', 'PC 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis=1)

print(pca.explained_variance_ratio_)

values = finalDf[['PC 1', 'PC 2']].to_numpy()

colors = ['red', 'green','blue','black']
kmeans = KMeans(n_clusters=3, random_state=0).fit(values)

'''
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('coefficient alpha1', fontsize=15)
ax.set_ylabel('coefficient alpha2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
for i in range(len(values)):
    ax.scatter(values[i][0], values[i][1], color=colors[kmeans.labels_[i]])
ax.grid()
plt.savefig("monaPC.svg",bbox_inches='tight')
plt.show()
'''
signal=np.mean(feat, axis=0)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(wav,signal,color='black')
for i in numbers:
    plt.scatter(wav[i],signal[i], color=colors[kmeans.labels_[int(numbers.index(i)*5)]])
ax.set_xlabel('wavelength (nm)', fontsize=15)
plt.savefig("monaSpectrum.svg",bbox_inches='tight')
plt.show()


fig=plt.figure()
angles=[2*np.pi*(i*10)/360 for i in range(37)]
ax1 = fig.add_subplot(projection='polar')
plt.plot(angles,pca.components_[0],label='PC 1')
plt.plot(angles,pca.components_[1],label='PC 2')
plt.yticks(fontsize=0)
plt.legend()
plt.show()

'''
### 3D plot

import plotly.graph_objects as go

x, y, z = finalDf['PC 1'], finalDf['PC 2'], finalDf['PC 3']
color=[colors[0]]*50+[colors[1]]*50+[colors[2]]*50

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=5,color='black')
)])

fig.update_layout(scene = dict(
                    xaxis_title='PC 1',
                    yaxis_title='PC 2',
                    zaxis_title='PC 3'),
                    margin=dict(r=0, b=0, l=0, t=0))
fig.write_html("test.html")
fig.show()

import webbrowser
url = "file:///C:/Users/morel/OneDrive/Bureau/Codes/test.html"
webbrowser.open(url,new=2)
'''