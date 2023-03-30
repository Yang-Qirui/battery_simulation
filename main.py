import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt

I_mag = 5.0
OCV_init = 4.0  # used for intial guess
Ri_init = 5e-2  # used for intial guess
R_busbar = 1.5e-3
R_connection = 1e-2
Np = 4
Ns = 1
Nbatt = Np * Ns
netlist = lp.setup_circuit(
    Np=Np, Ns=Ns, Rb=R_busbar, Rc=R_connection, Ri=Ri_init, V=OCV_init, I=I_mag
)

print(netlist)

experiment = pybamm.Experiment(
    [
        "Charge at 5 A for 30 minutes",
        "Rest for 15 minutes",
        "Discharge at 5 A for 30 minutes",
        "Rest for 15 minutes",
    ],
    period="10 seconds",
)

parameter_values = pybamm.ParameterValues("Chen2020")
SPMe = pybamm.models.full_battery_models.lithium_ion.SPMe()
# SPMe.variables.search("current")
#Battery voltage [V] Current [A]
output_variables = [
    "Battery voltage [V]",
    "Current [A]",
]

output = lp.solve(
    netlist=netlist,
    parameter_values=parameter_values,
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=0.5
)
 
lp.plot_output(output)
figures = plt.get_fignums() # 获取所有图形的编号

for i in figures:
    fig = plt.figure(i) # 获取当前编号对应的图形的引用
    fig.savefig('myplot{}.png'.format(i)) # 将图形保存到文件
