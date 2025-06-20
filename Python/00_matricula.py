from datetime import datetime

from feature_store.matricula.matricula import MatriculaClass

mtr = MatriculaClass()
mtr.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    mtr.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    mtr.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP MERGE PROCESS")
    mtr.silver_to_silver_eop()


def silver_to_gold():

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    mtr.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: WINDOW AGGREGATION FEATURE LEVEL EOP")
    mtr.gold_eop_to_window_aggregation()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO RUTINA MATRICULA")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA MATRICULA TERMINADO EN {(end_time-start_time)}")
