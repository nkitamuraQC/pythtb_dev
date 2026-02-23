from pythtb_respack import tb_model
import numpy as np

def group_velocity_1d(tbmodel: tb_model, klist: list) -> float:
    es = []
    for k in klist:
        kham = tbmodel._gen_ham(k_input=k)
        e, c = tbmodel._sol_ham(kham)
        es.append(e)
    es = np.array(es)
    v = np.gradient(es, klist)
    return v

def fermi_dirac_diff(tbmodel: tb_model, klist: list, mu: float, T: float) -> float:
    es = []
    for k in klist:
        kham = tbmodel._gen_ham(k_input=k)
        e, c = tbmodel._sol_ham(kham)
        es.append(e)
    es = np.array(es)
    beta = 1.0 / T
    x = beta * (es - mu)

    # オーバーフロー回避
    f = 1.0 / (np.exp(x) + 1.0)
    return -beta * f * (1.0 - f)

    
if __name__ == "__main__":
    from pythtb_respack import tb_model
    import matplotlib.pyplot as plt

    # 1D tight-binding model
    lat = [[1.0], [0.0]]
    orb = [[0.0], [0.0]]
    hopp = [[-1.0, 0, 0]]
    tbmodel = tb_model(1, 1, lat, orb, hopp)

    klist = np.linspace(-np.pi, np.pi, 100)
    v = group_velocity_1d(tbmodel, klist)
    plt.plot(klist, v)
    plt.xlabel("k")
    plt.ylabel("Group Velocity")
    plt.title("Group Velocity of 1D Tight-Binding Model")
    plt.grid()
    plt.show()