import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trans_mea import df_sort

fig = plt.figure(figsize=(12, 8))

Frequency = np.linspace(60000, 30000000, 501)


def impedance_vector(a, b):
    return a * b


def characteristic(a, b):
    return np.sqrt(a*b)


def gamma(short, open):
    x = np.sqrt(short/open)
    length = 11
    return 1/(2*length)*np.log((1 + x)/(1 - x))


# open
mea_open = pd.read_csv('C1XMEM03.CSV', usecols=[0], header=None)
IMPEDANCE_open = 10 ** (mea_open / 10) / 1000
IMPEDANCE_open = IMPEDANCE_open.values

arg_open = pd.read_csv('C1XMEM03.CSV', usecols=[1], header=None)
ARG_open = arg_open.values.tolist()
ARG_open_real = np.cos(np.radians(ARG_open))
ARG_open_imag = np.sin(np.radians(ARG_open))

open_real = impedance_vector(IMPEDANCE_open, ARG_open_real)
open_imag = impedance_vector(IMPEDANCE_open, ARG_open_imag)

open_complex = []
for i in range(len(mea_open)):
    open_complex.append(complex(open_real[i], open_imag[i]))


# short
mea_short = pd.read_csv('C1XMEM04.CSV', usecols=[0], header=None)
IMPEDANCE_short = 10 ** (mea_short / 10) / 1000
print(type(IMPEDANCE_short))

IMPEDANCE_short = IMPEDANCE_short.values


arg_short = pd.read_csv('C1XMEM04.CSV', usecols=[1], header=None)
ARG_short = arg_short.values.tolist()
ARG_short_real = np.cos(np.radians(ARG_short))
ARG_short_imag = np.sin(np.radians(ARG_short))


short_real = impedance_vector(IMPEDANCE_short, ARG_short_real)
short_imag = impedance_vector(IMPEDANCE_short, ARG_short_imag)


short_complex = []
for j in range(len(mea_open)):
    short_complex.append(complex(short_real[j], short_imag[j]))


characteristic_impedance = []
for k in range(len(mea_open)):
    characteristic_impedance.append(
        characteristic(open_complex[k], short_complex[k]))

characteristic_impedance_real = np.array(characteristic_impedance).real
characteristic_impedance_imag = np.array(characteristic_impedance).imag

Gamma = []
for l in range(len(mea_open)):
    Gamma.append(gamma(short_complex[l], open_complex[l]))

Gamma_real = np.array(Gamma).real
Gamma_imag = np.array(Gamma).imag

# 直線にする部分

gamma_imag = Gamma_imag

phase = []
for i in range(0, len(Gamma_imag)):
    phase.append(Gamma_imag[i])

Frequency_gamma = []
for i in range(0, len(Frequency)):
    Frequency_gamma.append(Frequency[i])

initial_point = phase[0]
start_point = []
end_point = []
final_point = phase[-1]

for j in range(0, len(Gamma_imag)):
    # out of range対策
    if j+1 == len(Gamma_imag):
        break
    # 位相の一番下のところ
    if phase[j] > phase[j+1]:
        start_point.append(phase[j+1])
        end_point.append(phase[j])

between_phase = phase[phase.index(initial_point): phase.index(start_point[0])]

# print(between_phase)
break_point = phase.index(start_point[0])
del Frequency_gamma[break_point]

for m in range(len(end_point) - 1):
    amount_of_change = start_point[m]
    between_start_end = phase[phase.index(
        start_point[m])+1: phase.index(end_point[m+1])+1]
    # between_start_end = np.multiply(between_start_end, )
    between_start_end = between_start_end + \
        between_phase[-1] + amount_of_change * -1
    between_phase.extend(between_start_end)
    # print(between_phase)

    break_point = phase.index(start_point[m+1])
    del Frequency_gamma[break_point]


amount_of_change = start_point[-1]
between_start_end_final = phase[phase.index(
    start_point[-1])+1: phase.index(final_point)+1]
# between_start_end_final = np.multiply(between_start_end_final, ).tolist()
between_start_end_final = between_start_end_final + \
    between_phase[-1] + amount_of_change*-1
between_phase.extend(between_start_end_final)


# add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Real-Impedance [Ω]")

# ax1.set_title("0pen-short(real)")

ax2 = fig.add_subplot(2, 3, 2)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Imag-Impedance [Ω]")
# ax2.set_title("0pen-short(imag)")

ax3 = fig.add_subplot(2, 3, 3)
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Real-Charcterristic Impedance [Ω]")
ax3.set_title("Characteristic Impedance")

ax4 = fig.add_subplot(2, 3, 4)
ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Propagation Constant [Np/m]")
# ax4.set_title("gamma(real)")

ax5 = fig.add_subplot(2, 3, 5)
ax5.set_xlabel("Frequency [Hz]")
ax5.set_ylabel("Phase Constant [rad/m]")

ax6 = fig.add_subplot(2, 3, 6)
ax6.set_xlabel("Frequency [Hz]")
ax6.set_ylabel("phase-linear [rad/m]")
ax6.set_title("Linear Gamma-Imag")


c1, c2, c3, c4 = "blue", "green", "red", "black"      # 各プロットの色
# 各ラベル
l1, l2, l3, l4, l5, l6, l7 = "open-real", "open-imag", "short-real", "short-imag", "gamma(real)", "gamma(imag)", "characteristic_impedance_real"


range = (30000000 - 60000)/6
ax1.plot(Frequency, open_real, color=c3, label=l1)
ax1.plot(Frequency, open_imag, color=c1, label=l2)
ax2.plot(Frequency, short_real, color=c3, label=l3)
ax2.plot(Frequency, short_imag, color=c1, label=l4)
ax3.plot(Frequency, characteristic_impedance_real, color=c3)
ax4.plot(Frequency, Gamma_real, color=c2, label=l5)
ax4.plot(Frequency, df_sort, color=c1, label="propa")
ax5.plot(Frequency, Gamma_imag, color=c4, label=l6)
ax6.plot(Frequency_gamma, between_phase, color="red")


ax1.legend(loc='lower left')  # 凡例
ax2.legend(loc='lower left')  # 凡例
ax3.legend(loc='upper right')  # 凡例
ax4.legend(loc='lower center')  # 凡例
ax5.legend(loc='lower center')  # 凡例


ax1.set_xlim([60000, 30000000])
ax1.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 +
               3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

ax2.set_xlim([60000, 30000000])
ax2.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 +
               3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

ax3.set_xlim([60000, 30000000])
ax3.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 +
               3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

ax4.set_xlim([60000, 30000000])
ax4.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 +
               3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

ax5.set_xlim([60000, 30000000])
ax5.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 +
               3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

ax6.set_xlim([60000, 30000000])
ax6.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 +
               3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

fig.tight_layout()  # レイアウトの設定
# plt.show()

fig.savefig("./figure/RG58A.U.pdf")
