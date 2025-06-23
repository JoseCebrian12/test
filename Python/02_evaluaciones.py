from datetime import datetime

from feature_store.evaluaciones.evaluaciones import EvalClass

eval = EvalClass()
eval.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    eval.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    eval.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP PROCESS")
    eval.silver_to_silver_eop()

    print("\nSILVER_TO_SILVER: DATE PROCESS")
    eval.silver_to_silver_date()


def silver_to_gold():

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    eval.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP AVANCE CURSO")
    eval.silver_to_gold_eop_ac()

    print("\nSILVER_TO_GOLD: WINDOW AGGREGATION FEATURE LEVEL EOP")
    eval.gold_eop_to_window_aggregation()

    print("\nSILVER_TO_GOLD: DATE PROCESS")
    eval.silver_to_gold_date()

    print("\nSILVER_TO_GOLD: DATE PROCESS AVANCE CURSO")
    eval.silver_to_gold_date_ac()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO EVALUACIONES")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA EVALUACIONES TERMINADO EN {end_time-start_time}")
