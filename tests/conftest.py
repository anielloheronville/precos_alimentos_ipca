import os
import sys

import pytest

# Garante que os workers do Spark usem o mesmo Python do pytest (evita o "python3" do
# Windows Store ou outra instalação no PATH).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def _garantir_spark_home_sem_acentos() -> None:
    """No Windows, os scripts .cmd do Spark corrompem caminhos com acento (ex.: "preços")
    e a JVM falha com `ClassNotFoundException: SparkSubmit`. Quando o pyspark está num
    caminho assim, usa o nome curto 8.3 do Windows (só ASCII) como SPARK_HOME."""
    if os.name != "nt" or "SPARK_HOME" in os.environ:
        return

    import ctypes

    import pyspark

    spark_home = os.path.dirname(pyspark.__file__)
    if spark_home.isascii():
        return

    buffer = ctypes.create_unicode_buffer(32768)
    tamanho = ctypes.windll.kernel32.GetShortPathNameW(spark_home, buffer, len(buffer))
    if tamanho and buffer.value.isascii():
        os.environ["SPARK_HOME"] = buffer.value
        return

    raise RuntimeError(
        f"O pyspark está num caminho com acentos ({spark_home}) e o Windows não tem nome curto "
        "8.3 para ele. Crie o ambiente virtual num caminho sem acentos ou defina SPARK_HOME "
        "apontando para uma pasta sem acentos que leve ao pacote pyspark."
    )


@pytest.fixture(scope="session")
def spark():
    _garantir_spark_home_sem_acentos()

    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-soja-alimentos")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
