from datetime import datetime

from feature_store.retiro_ciclo.retiro_ciclo import RetiroCicloClass

ret_ciclo = RetiroCicloClass()
ret_ciclo.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    ret_ciclo.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    ret_ciclo.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP PROCESS")
    ret_ciclo.silver_to_silver_eop()


def silver_to_gold():

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    ret_ciclo.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: WINDOW AGGREGATION FEATURE LEVEL EOP")
    ret_ciclo.gold_eop_to_window_aggregation()

    print("\nSILVER_TO_GOLD: DATE PROCESS")
    ret_ciclo.silver_to_gold_date()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO RETIRO CICLO")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA RETIRO CICLO TERMINADO EN {end_time-start_time}")
