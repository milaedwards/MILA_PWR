# run.py
from config import Config
from final_sim.main import run
from final_sim.stubs import ReactorSimulator, PressurizerPI, SG, TurbineCondenser

if __name__ == "__main__":
    cfg = Config()
    reactor = ReactorSimulator(Tc_init=cfg.T_COLD_INIT_K, P_turb_init=1.0, control_mode="auto")
    pressurizer = PressurizerPI(P_set_Pa=cfg.P_PRI_SET)
    steamgen = SG()
    turbine = TurbineCondenser()
    run(reactor=reactor, pressurizer=pressurizer, steamgen=steamgen, turbine=turbine, cfg=cfg, early_stop=True, csv_out=False)
