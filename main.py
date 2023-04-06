import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import os
import json

'''
电极厚度：更改正负电极的厚度 (Positive electrode thickness [m] 和 Negative electrode thickness [m]) 可以影响电池的容量和功率性能。减小电极厚度会降低电池的容量，可能导致电池过早失效。
电极导电率：降低正负电极的导电率 (Positive electrode conductivity [S.m-1] 和 Negative electrode conductivity [S.m-1]) 可能导致电池内部电阻增加，从而降低电池的输出功率。
电池的上下限电压：更改电池的上限电压 (Upper voltage cut-off [V]) 和下限电压 (Lower voltage cut-off [V]) 可以模拟过充或过放的故障情况。过充或过放可能导致电池性能下降或出现安全隐患。
电解质浓度：更改电解质初始浓度 (Initial concentration in electrolyte [mol.m-3]) 可以影响离子传输速率。过低的浓度可能导致电池传输性能下降。
SEI层厚度：增加初始 SEI（固体电解质界面）层的厚度 (Initial inner SEI thickness [m] 和 Initial outer SEI thickness [m]) 可以模拟 SEI 层过厚导致的电池性能下降。
温度：更改初始温度 (Initial temperature [K]) 可以模拟电池在不同温度下的工作情况。较高或较低的温度可能导致电池性能下降。
极板间隙：增加分隔膜厚度 (Separator thickness [m]) 可能导致电池内部电阻增加，从而降低电池的输出功率。
'''

def sei_degration_sim(parameter_values=None):
    model = pybamm.lithium_ion.SPM(options={
            # "operating mode":"CCCV",
            "SEI": "ec reaction limited",
            "SEI film resistance": "distributed",
            "SEI porosity change": "true",
            "calculate discharge energy": "true",
            # "loss of active material":"reaction-driven",
            "lithium plating porosity change":"true",
            # "lithium plating":"reversible"
        })
    model = lp.add_events_to_model(model)
    solver = pybamm.CasadiSolver(mode="safe")
    return pybamm.Simulation(
    model=model,
    parameter_values=parameter_values,
    solver=solver
    )

def conf_ambient_temp_input(Np,Ns,_type='normal'):
    def normal_temp():
        return {"Ambient temperature [K]": (np.ones((Np,Ns)) * 298.15).flatten()}
    def abnormal_temp():
        # 创建一个值为 1 的 m x n 矩阵
        matrix = np.ones((Np, Ns))
        if 0.1 * Np * Ns >= 2:
            k = k = np.random.randint(1, int(0.1*Np*Ns) + 1)
        else: k = 1

        # 随机选择 k 个不重复的位置
        num_cells = Np * Ns
        indices = np.random.choice(num_cells, k, replace=False)

        # 在选定的位置上设置值为 0.8 到 1.2 之间的随机数
        for idx in indices:
            row, col = divmod(idx, Ns)  # 计算行和列的索引
            bias = np.random.randint(-2,2)
            while bias == 0:
               bias = np.random.randint(-2,2)
            matrix[row, col] = 1 + bias / 10

        matrix *= 298.1
        
        return {"Ambient temperature [K]": matrix.flatten()}

    switch_case = {
        'normal': normal_temp,
        'abnormal': abnormal_temp
    }

    func_to_call = switch_case.get(_type)
    assert func_to_call != None
    return func_to_call()

def conf_resistance_input(Np,Ns,_type='normal'):
    def normal_temp():
        return {"Ambient temperature [K]": (np.ones((Np,Ns)) * 298.15).flatten()}
    def abnormal_temp():
        # 创建一个值为 1 的 m x n 矩阵
        matrix = np.ones((Np, Ns))
        if 0.1 * Np * Ns >= 2:
            k = k = np.random.randint(1, int(0.1*Np*Ns) + 1)
        else: k = 1

        # 随机选择 k 个不重复的位置
        num_cells = Np * Ns
        indices = np.random.choice(num_cells, k, replace=False)

        # 在选定的位置上设置值为 0.8 到 1.2 之间的随机数
        for idx in indices:
            row, col = divmod(idx, Ns)  # 计算行和列的索引
            bias = np.random.randint(-2,2)
            while bias == 0:
               bias = np.random.randint(-2,2)
            matrix[row, col] = 1 + bias / 10

        matrix *= 298.1
        
        return {"Ambient temperature [K]": matrix.flatten()}

    switch_case = {
        'normal': normal_temp,
        'abnormal': abnormal_temp
    }

    func_to_call = switch_case.get(_type)
    assert func_to_call != None
    return func_to_call()

def simulate(args):

    I_mag = 5.0
    OCV_init = 4.0  # used for intial guess
    Ri_init = 5e-2  # used for intial guess
    R_busbar = 1.5e-3
    R_connection = 1e-2
    Np = args.Np
    Ns = args.Ns
    Nbatt = Np * Ns
    netlist = lp.setup_circuit(
        Np=Np, Ns=Ns, Rb=R_busbar, Rc=R_connection, Ri=Ri_init, V=OCV_init, I=I_mag
    )
    lp.draw_circuit(netlist, cpt_size=1.0, dpi=200, node_spacing=2.5, scale=0.75)
    exp_type = args.type

    parameter_values = pybamm.ParameterValues("Chen2020")
    
    # print(parameter_values)

    experiment = pybamm.Experiment(
        [
        (
            "Discharge at 5 A for 1000 s or until 3.3 V",
            "Rest for 1000 s",
            "Charge at 5 A for 1000 s or until 4.1 V",
            "Rest for 1000 s",
        )
        ] * 3,
        period="100 s",
    )
    output_variables = [
        'X-averaged total SEI thickness [m]',
        'Loss of capacity to SEI [A.h]',
        # 'Discharge capacity [A.h]'
    ]

    # ambient temperature
    if 'T' in args.params:
        normal_ambient_temp = conf_ambient_temp_input(Np,Ns,exp_type)
        parameter_values.update({"Ambient temperature [K]":"[input]"})
    
    if args.save_params:
        with open('params.json', 'w') as f:
            str_param_values = {}
            for k,v in parameter_values.items():
                str_param_values[k] = str(v)
            json.dump(str_param_values, f)
            f.close()

    output,rm = lp.solve(
        netlist=netlist,
        parameter_values=parameter_values,
        experiment=experiment,
        output_variables=output_variables,
        initial_soc=0.5,
        external_variables=normal_ambient_temp,
        sim_func=sei_degration_sim,
    )
    print(rm.all_output)

    return
    lp.plot_output(output)
    figures = plt.get_fignums() # 获取所有图形的编号

    if not os.path.exists('plots'):
        os.mkdir('plots')
    now = datetime.datetime.now()
    os.mkdir(f"plots/{now}")
    for i in figures:
        fig = plt.figure(i) # 获取当前编号对应的图形的引用
        fig.savefig(f'plots/{now}/myplot{i}.png') # 将图形保存到文件

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-type",help="The simulation type for battery. Eg.[normal, abnormal]")
    arg_parser.add_argument("-params",help="The params array to adjust. Eg.[T(temperature),R(resistance)]",nargs='+')
    arg_parser.add_argument("-Np",help="The parallel number of batteries",default=4,type=int)
    arg_parser.add_argument("-Ns",help="The sequential number of batteries",default=1,type=int)
    arg_parser.add_argument("-save_params",help="Save parameters values or not",default=False,action='store_true')
    # Total batteries = Np * Ns
    args = arg_parser.parse_args()
    simulate(args)