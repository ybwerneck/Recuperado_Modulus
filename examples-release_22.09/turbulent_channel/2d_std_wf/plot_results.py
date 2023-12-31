import numpy as np
import matplotlib.pyplot as plt
from modulus.utils.io import csv_to_dict

# path for checkpoint
checkpoint = "./outputs/re590_k_ep/network_checkpoint/"

# read data to compute u_tau
data = np.load(checkpoint + "inferencers/inf_wf.npz", allow_pickle=True)
data = np.atleast_1d(data.f.arr_0)[0]
u_tau = np.mean(data["u_tau"])

# read data to plot profiles
interior_data = np.load(checkpoint + "inferencers/inf_interior.npz", allow_pickle=True)
interior_data = np.atleast_1d(interior_data.f.arr_0)[0]
y = interior_data["y"]
u = interior_data["u"]
k = interior_data["k"]

nu = 1 / 590
u_plus = u / u_tau
y_plus = (1 - np.abs(y)) * u_tau / nu
k_plus = k / u_tau / u_tau
y = 1 - np.abs(y)

# read validation data
# Fluent data from Turbulence lecture notes: Gianluca Iaccarino: https://web.stanford.edu/class/me469b/handouts/turbulence.pdf
# DNS data from Moser et al.: https://aip.scitation.org/doi/10.1063/1.869966

mapping = {"u+": "u_plus", "y+": "y_plus"}
u_dns_data = csv_to_dict("../validation_data/re590-moser-dns-u_plus.csv", mapping)
u_fluent_gi_data = csv_to_dict("../validation_data/re590-gi-fluent-u_plus.csv", mapping)

mapping = {"k+": "k_plus", "y/2H": "y"}
k_dns_data = csv_to_dict("../validation_data/re590-moser-dns-k_plus.csv", mapping)
k_fluent_gi_data = csv_to_dict("../validation_data/re590-gi-fluent-k_plus.csv", mapping)

fig, ax = plt.subplots(2, figsize=(4.5, 9))
ax[0].scatter(y, k_plus, label="Modulus")
ax[0].scatter(k_dns_data["y"], k_dns_data["k_plus"], label="DNS: Moser")
ax[0].scatter(k_fluent_gi_data["y"], k_fluent_gi_data["k_plus"], label="Fluent: GI")
ax[0].set(title="TKE: u_tau=" + str(round(u_tau, 3)))
ax[0].set(xlabel="y", ylabel="k+")
ax[0].legend()

ax[1].scatter(y_plus, u_plus, label="Modulus")
ax[1].scatter(u_dns_data["y_plus"], u_dns_data["u_plus"], label="DNS: Moser")
ax[1].scatter(
    u_fluent_gi_data["y_plus"], u_fluent_gi_data["u_plus"], label="Fluent: GI"
)
ax[1].set_xscale("log")
ax[1].set(title="U+: u_tau=" + str(round(u_tau, 3)))
ax[1].set(xlabel="y+", ylabel="u+")
ax[1].legend()

plt.tight_layout()
plt.savefig("results_k_ep.png")
