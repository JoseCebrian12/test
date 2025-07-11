from datetime import datetime

from feature_store.retiro_curso.retiro_curso import RetiroCursoClass

ret_curso = RetiroCursoClass()
ret_curso.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    ret_curso.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    ret_curso.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP PROCESS")
    ret_curso.silver_to_silver_eop()


def silver_to_gold():

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    ret_curso.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: WINDOW AGGREGATION FEATURE LEVEL EOP")
    ret_curso.gold_eop_to_window_aggregation()

    print("\nSILVER_TO_GOLD: DATE PROCESS")
    ret_curso.silver_to_gold_date()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO RETIRO CURSO")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA RETIRO CURSO TERMINADO EN {end_time-start_time}")
