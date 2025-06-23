from datetime import datetime

from feature_store.cursos_matricula.cursos_matricula import CursoMatricula

cur_mtr = CursoMatricula()
cur_mtr.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    cur_mtr.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    cur_mtr.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP MERGE PROCESS")
    cur_mtr.silver_to_silver_eop()


def silver_to_gold():

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    cur_mtr.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: WINDOW AGGREGATION FEATURE LEVEL EOP")
    cur_mtr.gold_eop_to_window_aggregation()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO CURSOS MATRICULA")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA CURSOS MATRICULA TERMINADO EN {end_time-start_time}")
