#!/usr/bin/env python3
"""
Transcribe en paralelo todos los PDF del directorio actual usando Gemini.

Instalación:
    uv sync

Configuración:
    Copia .env.example como .env y completa GEMINI_API_KEY.

Ejecución:
    python main.py

Variables opcionales:
    GEMINI_MODEL=gemini-3.5-flash-lite
    GEMINI_MAX_WORKERS=15
    GEMINI_MAX_RETRIES=4
    GEMINI_MAX_PAGES_PER_CHUNK=30
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter


# Carga el .env situado junto a este script, independientemente del directorio
# desde el que se ejecute el programa.
load_dotenv(Path(__file__).resolve().parent / ".env")


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_WORKERS = min(int(os.getenv("GEMINI_MAX_WORKERS", "15")), 15)
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
MAX_PAGES_PER_CHUNK = int(os.getenv("GEMINI_MAX_PAGES_PER_CHUNK", "30"))

SOURCE_DIR = Path.cwd()
OUTPUT_DIR = SOURCE_DIR / "markdown"

POLL_INTERVAL_SECONDS = 2
FILE_PROCESSING_TIMEOUT_SECONDS = 10 * 60

PROMPT = """
Actúa exclusivamente como un transcriptor jurídico documental de máxima
precisión.

Tu tarea es transcribir ÍNTEGRAMENTE el documento PDF adjunto a Markdown,
desde la primera hasta la última página.

REGLAS OBLIGATORIAS:

1. Transcribe absolutamente todo el contenido visible del documento, sin
   resumir, interpretar, corregir, modernizar, completar ni omitir nada.

2. Conserva exactamente:
   - palabras;
   - ortografía original;
   - errores tipográficos;
   - mayúsculas y minúsculas;
   - números;
   - fechas;
   - cantidades;
   - monedas;
   - nombres propios;
   - referencias;
   - numeración de cláusulas;
   - notas al pie;
   - encabezados;
   - pies de página;
   - sellos;
   - firmas legibles;
   - anotaciones manuscritas legibles;
   - marcas, códigos y números de radicación.

3. No corrijas errores del documento. No inventes texto ilegible. Cuando un
   fragmento realmente no pueda leerse, escribe exactamente:

   [ILEGIBLE]

   Cuando solo una parte sea ilegible, conserva el texto legible e inserta
   [ILEGIBLE] únicamente en la posición correspondiente.

4. Mantén el orden de lectura y la estructura lógica del original. Usa
   Markdown para representar:
   - títulos y subtítulos;
   - párrafos;
   - listas;
   - tablas;
   - citas;
   - secciones;
   - saltos de página.

5. Representa cada cambio de página mediante una línea independiente con este
   formato:

   <!-- Página N -->

   donde N es el número físico dentro del documento original, según el
   intervalo de páginas indicado al final de estas instrucciones.

6. Para las tablas:
   - transcribe todas las celdas;
   - conserva filas y columnas;
   - usa tablas Markdown cuando sea posible;
   - si una tabla es demasiado compleja, represéntala con HTML válido dentro
     del Markdown;
   - no conviertas una tabla en un resumen narrativo.

7. Examina también las imágenes, páginas escaneadas, fotografías, sellos,
   membretes, diagramas, capturas y elementos gráficos. Transcribe todo texto
   relevante que aparezca dentro de ellos.

8. Cuando una imagen aporte información jurídica o documental relevante que
   no pueda representarse solo mediante texto, añade una descripción objetiva
   y estrictamente factual con este formato:

   [Descripción de imagen: ...]

   No especules sobre información que no sea visible.

9. En firmas y sellos:
   - transcribe cualquier texto legible;
   - no identifiques personas por inferencia;
   - usa [Firma ilegible] cuando corresponda;
   - usa [Sello ilegible] cuando corresponda.

10. No agregues introducciones, explicaciones, conclusiones, advertencias,
    comentarios editoriales ni bloques de código.

11. No encierres la respuesta entre ```markdown ni entre otras cercas de
    código.

12. La respuesta debe contener exclusivamente la transcripción completa en
    Markdown.

13. Antes de finalizar, verifica internamente que procesaste todas las páginas
    y que no omitiste anexos, tablas, notas marginales, reversos, páginas en
    blanco identificables ni texto contenido en imágenes.

Comienza directamente con la transcripción del documento.
""".strip()


class ResponseTruncatedError(RuntimeError):
    """Indica que Gemini agotó el presupuesto de tokens de salida."""


@dataclass(frozen=True)
class PageOmission:
    """Describe un intervalo que no pudo transcribirse."""

    start_page: int
    end_page: int
    error: Exception


@dataclass(frozen=True)
class PageRangeResult:
    """Resultado completo o parcial de un intervalo de páginas."""

    markdown: str
    omissions: tuple[PageOmission, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    """Resultado final de un PDF, incluso si contiene páginas omitidas."""

    output_path: Path
    omissions: tuple[PageOmission, ...] = ()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def create_client() -> genai.Client:
    """
    Crea un cliente independiente por worker.

    Evita compartir conexiones o estado interno entre threads.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No se encontró GEMINI_API_KEY ni GOOGLE_API_KEY en el entorno."
        )

    return genai.Client(api_key=api_key)


def normalize_state(state: Any) -> str:
    """
    Convierte el estado devuelto por el SDK en un texto comparable.

    Dependiendo de la versión del SDK, puede ser un string o un enum.
    """
    if state is None:
        return ""

    value = getattr(state, "value", state)
    normalized = str(value).upper()

    if "." in normalized:
        normalized = normalized.rsplit(".", maxsplit=1)[-1]

    return normalized


def wait_until_file_is_active(
    client: genai.Client,
    uploaded_file: Any,
) -> Any:
    """
    Espera hasta que Gemini termine de procesar el PDF subido.
    """
    file_name = getattr(uploaded_file, "name", None)

    if not file_name:
        raise RuntimeError("La Files API no devolvió el nombre del archivo.")

    deadline = time.monotonic() + FILE_PROCESSING_TIMEOUT_SECONDS

    while True:
        remote_file = client.files.get(name=file_name)
        state = normalize_state(getattr(remote_file, "state", None))

        if state in {"ACTIVE", "SUCCEEDED", "READY", ""}:
            return remote_file

        if state in {"FAILED", "ERROR", "CANCELLED"}:
            error = getattr(remote_file, "error", None)
            raise RuntimeError(
                f"Gemini no pudo procesar el archivo. Estado={state}; "
                f"detalle={error!r}"
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Se agotó el tiempo esperando el procesamiento del PDF "
                f"remoto: {file_name}"
            )

        logging.info(
            "El archivo remoto sigue procesándose: %s (%s)",
            file_name,
            state or "estado desconocido",
        )
        time.sleep(POLL_INTERVAL_SECONDS)


def extract_response_text(response: Any) -> str:
    """
    Extrae y valida la respuesta final no streaming.
    """
    text = getattr(response, "text", None)

    if not text or not text.strip():
        finish_reason = get_finish_reason(response)

        if finish_reason == "RECITATION":
            raise RuntimeError(
                "Gemini bloqueó la respuesta por restricciones de recitación "
                "o copyright. No es un error de carga ni de límite de tokens. "
                f"finish_reason={finish_reason}"
            )

        prompt_feedback = getattr(response, "prompt_feedback", None)
        candidates = getattr(response, "candidates", None)

        raise RuntimeError(
            "Gemini devolvió una respuesta sin texto. "
            f"prompt_feedback={prompt_feedback!r}; "
            f"candidates={candidates!r}"
        )

    return text.strip() + "\n"


def get_finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []

    if not candidates:
        return ""

    finish_reason = getattr(candidates[0], "finish_reason", None)

    if finish_reason is None:
        return ""

    value = getattr(finish_reason, "value", finish_reason)
    return str(value).upper()


def ensure_response_not_truncated(response: Any) -> None:
    """
    No guarda silenciosamente una transcripción que la API marque como
    truncada por límite de tokens.
    """
    finish_reason = get_finish_reason(response)

    truncation_markers = {
        "MAX_TOKENS",
        "FINISH_REASON_MAX_TOKENS",
        "LENGTH",
    }

    if any(marker in finish_reason for marker in truncation_markers):
        raise ResponseTruncatedError(
            "La respuesta alcanzó el límite máximo de tokens. "
            "La transcripción probablemente está incompleta. "
            f"finish_reason={finish_reason}"
        )


def safe_output_name(pdf_path: Path) -> str:
    """
    Produce un nombre seguro y determinista.

    Ejemplo:
        Contrato final.PDF -> Contrato final.md
    """
    stem = pdf_path.stem.strip() or "documento"
    stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", stem)
    return f"{stem}.md"


def safe_upload_display_name(pdf_path: Path) -> str:
    """
    Convierte el nombre local en un nombre ASCII apto para cabeceras HTTP.

    httpx no puede codificar tildes o caracteres como ñ en los valores de
    cabecera que usa la Files API para el nombre visible del archivo.
    """
    display_name = unicodedata.normalize("NFKD", pdf_path.name)
    display_name = display_name.encode("ascii", "ignore").decode("ascii")
    display_name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", display_name)
    return display_name or "documento.pdf"


def atomic_write_text(destination: Path, content: str) -> None:
    """
    Evita dejar archivos parciales si el proceso se interrumpe durante
    la escritura.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def is_retryable_exception(error: Exception) -> bool:
    """
    Clasificación conservadora de errores transitorios.

    El SDK puede envolver errores HTTP en distintas clases según su versión,
    por lo que también se inspecciona el mensaje.
    """
    message = str(error).lower()

    non_retryable_markers = (
        "api key",
        "permission denied",
        "invalid argument",
        "unsupported",
        "not found",
        "response reached",
        "respuesta alcanzó",
        "probably incomplete",
        "probablemente está incompleta",
    )

    if any(marker in message for marker in non_retryable_markers):
        return False

    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "resource exhausted",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "internal error",
    )

    return any(marker in message for marker in retryable_markers)


def delete_remote_file(client: genai.Client, uploaded_file: Any) -> None:
    """
    Elimina anticipadamente el PDF remoto.

    Aunque Files API lo conserva temporalmente, no es necesario dejarlo
    almacenado una vez terminada la transcripción.
    """
    file_name = getattr(uploaded_file, "name", None)

    if not file_name:
        return

    try:
        client.files.delete(name=file_name)
        logging.info("Archivo remoto eliminado: %s", file_name)
    except Exception as error:
        # La transcripción local ya puede ser válida; un fallo de limpieza
        # no debe convertir el trabajo completo en fallido.
        logging.warning(
            "No se pudo eliminar el archivo remoto %s: %s",
            file_name,
            error,
        )


def build_chunk_prompt(
    start_page: int,
    end_page: int,
    total_pages: int,
) -> str:
    """Añade al prompt la numeración real del fragmento enviado."""
    return (
        f"{PROMPT}\n\n"
        "INSTRUCCIÓN OBLIGATORIA DE NUMERACIÓN:\n"
        f"El PDF adjunto contiene las páginas físicas {start_page} a "
        f"{end_page} de un documento original de {total_pages} páginas. "
        f"Su primera página debe marcarse como <!-- Página {start_page} -->, "
        "no como página 1. Continúa la numeración consecutivamente hasta "
        f"<!-- Página {end_page} -->."
    )


def write_pdf_chunk(
    reader: PdfReader,
    start_index: int,
    end_index: int,
    destination: Path,
) -> None:
    """Escribe un intervalo de páginas, con índices de inicio inclusivo."""
    writer = PdfWriter()

    for page_index in range(start_index, end_index):
        writer.add_page(reader.pages[page_index])

    with destination.open("wb") as chunk_file:
        writer.write(chunk_file)


def combine_markdown(parts: list[str]) -> str:
    """Une fragmentos sin perder el salto que separa sus páginas."""
    return "\n\n".join(part.strip() for part in parts if part.strip()) + "\n"


def incomplete_text_marker(start_page: int, end_page: int) -> str:
    """Genera una advertencia visible en el lugar del intervalo omitido."""
    return (
        "TEXTO INCOMPLETO, NO SE INCLUYEN LAS PÁGINAS "
        f"{start_page} A LA {end_page}"
    )


def transcribe_chunk_once(
    upload_path: Path,
    original_pdf_path: Path,
    start_page: int,
    end_page: int,
    total_pages: int,
) -> str:
    """Sube y transcribe un intervalo de páginas en una sola petición."""
    client = create_client()
    uploaded_file = None
    page_range = f"páginas {start_page}-{end_page}/{total_pages}"

    try:
        logging.info("Subiendo: %s (%s)", original_pdf_path.name, page_range)

        # Al recibir una ruta, el SDK coloca el nombre original en la cabecera
        # X-Goog-Upload-File-Name, que httpx no puede codificar si contiene
        # tildes o ñ. Un stream binario evita esa cabecera problemática.
        with upload_path.open("rb") as pdf_file:
            uploaded_file = client.files.upload(
                file=pdf_file,
                config=types.UploadFileConfig(
                    mime_type="application/pdf",
                    display_name=safe_upload_display_name(original_pdf_path),
                ),
            )

        remote_file = wait_until_file_is_active(client, uploaded_file)

        logging.info(
            "Transcribiendo: %s (%s)",
            original_pdf_path.name,
            page_range,
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                remote_file,
                build_chunk_prompt(start_page, end_page, total_pages),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=65_536,
                thinking_config=types.ThinkingConfig(
                    thinking_level="MINIMAL",
                ),
            ),
        )

        ensure_response_not_truncated(response)
        return extract_response_text(response)

    finally:
        if uploaded_file is not None:
            delete_remote_file(client, uploaded_file)

        client.close()


def transcribe_chunk_with_retries(
    upload_path: Path,
    original_pdf_path: Path,
    start_page: int,
    end_page: int,
    total_pages: int,
) -> str:
    """Reintenta errores transitorios de un fragmento."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return transcribe_chunk_once(
                upload_path,
                original_pdf_path,
                start_page,
                end_page,
                total_pages,
            )

        except Exception as error:
            last_attempt = attempt == MAX_RETRIES

            if last_attempt or not is_retryable_exception(error):
                raise

            delay = min(60.0, (2 ** (attempt - 1)) + random.uniform(0, 1.5))

            logging.warning(
                "Intento %d/%d falló para %s (páginas %d-%d): %s. "
                "Reintentando en %.1f segundos.",
                attempt,
                MAX_RETRIES,
                original_pdf_path.name,
                start_page,
                end_page,
                error,
                delay,
            )

            time.sleep(delay)

    raise AssertionError("Bucle de reintentos terminado inesperadamente.")


def transcribe_page_range(
    reader: PdfReader,
    original_pdf_path: Path,
    start_index: int,
    end_index: int,
    total_pages: int,
    temporary_dir: Path,
) -> PageRangeResult:
    """
    Transcribe un intervalo, lo subdivide si agota los tokens y conserva
    una marca explícita si un subintervalo no puede procesarse.
    """
    start_page = start_index + 1
    end_page = end_index

    try:
        if start_index == 0 and end_index == total_pages:
            upload_path = original_pdf_path
        else:
            upload_path = temporary_dir / f"pages_{start_page}_{end_page}.pdf"
            write_pdf_chunk(reader, start_index, end_index, upload_path)

        markdown = transcribe_chunk_with_retries(
            upload_path,
            original_pdf_path,
            start_page,
            end_page,
            total_pages,
        )
        return PageRangeResult(markdown=markdown)
    except ResponseTruncatedError:
        page_count = end_index - start_index

        if page_count <= 1:
            single_page_error = RuntimeError(
                f"La página {start_page} por sí sola supera el límite de "
                "salida del modelo."
            )
            logging.error(
                "No se pudo transcribir la página %d de %s: %s",
                start_page,
                original_pdf_path.name,
                single_page_error,
            )
            return PageRangeResult(
                markdown=incomplete_text_marker(start_page, end_page),
                omissions=(
                    PageOmission(start_page, end_page, single_page_error),
                ),
            )

        midpoint = start_index + page_count // 2
        logging.warning(
            "%s agotó los tokens en las páginas %d-%d; "
            "se dividirá en %d-%d y %d-%d.",
            original_pdf_path.name,
            start_page,
            end_page,
            start_page,
            midpoint,
            midpoint + 1,
            end_page,
        )

        left = transcribe_page_range(
            reader,
            original_pdf_path,
            start_index,
            midpoint,
            total_pages,
            temporary_dir,
        )
        right = transcribe_page_range(
            reader,
            original_pdf_path,
            midpoint,
            end_index,
            total_pages,
            temporary_dir,
        )
        return PageRangeResult(
            markdown=combine_markdown([left.markdown, right.markdown]),
            omissions=left.omissions + right.omissions,
        )
    except Exception as error:
        logging.error(
            "No se pudieron transcribir las páginas %d-%d de %s: %s: %s",
            start_page,
            end_page,
            original_pdf_path.name,
            type(error).__name__,
            error,
        )
        return PageRangeResult(
            markdown=incomplete_text_marker(start_page, end_page),
            omissions=(PageOmission(start_page, end_page, error),),
        )


def transcribe_with_retries(pdf_path: Path) -> TranscriptionResult:
    """Divide, transcribe y combina un documento completo."""
    output_path = OUTPUT_DIR / safe_output_name(pdf_path)

    with pdf_path.open("rb") as source_file:
        reader = PdfReader(source_file)
        total_pages = len(reader.pages)

        if total_pages < 1:
            raise RuntimeError(f"El PDF no contiene páginas: {pdf_path.name}")

        ranges = [
            (start_index, min(start_index + MAX_PAGES_PER_CHUNK, total_pages))
            for start_index in range(0, total_pages, MAX_PAGES_PER_CHUNK)
        ]

        if len(ranges) > 1:
            logging.info(
                "%s tiene %d páginas; se procesará inicialmente en %d "
                "fragmentos de hasta %d páginas.",
                pdf_path.name,
                total_pages,
                len(ranges),
                MAX_PAGES_PER_CHUNK,
            )

        parts: list[str] = []
        omissions: list[PageOmission] = []

        with tempfile.TemporaryDirectory(prefix="gemini-pdf-") as directory:
            temporary_dir = Path(directory)

            for start_index, end_index in ranges:
                result = transcribe_page_range(
                    reader,
                    pdf_path,
                    start_index,
                    end_index,
                    total_pages,
                    temporary_dir,
                )
                parts.append(result.markdown)
                omissions.extend(result.omissions)

                # Guarda un checkpoint después de cada fragmento inicial. Si
                # una ejecución se interrumpe más adelante, lo ya transcrito
                # permanece disponible.
                atomic_write_text(output_path, combine_markdown(parts))

    relative_output = output_path.relative_to(SOURCE_DIR)

    if omissions:
        logging.warning(
            "Transcripción incompleta: %s -> %s. Se omitieron %d intervalo(s).",
            pdf_path.name,
            relative_output,
            len(omissions),
        )
    else:
        logging.info(
            "Completado: %s -> %s",
            pdf_path.name,
            relative_output,
        )

    return TranscriptionResult(output_path, tuple(omissions))


def find_pdfs(directory: Path) -> list[Path]:
    """
    Lista únicamente los PDF ubicados directamente en el directorio actual.

    La comparación de extensión no distingue mayúsculas de minúsculas.
    """
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )


def main() -> int:
    configure_logging()

    if MAX_WORKERS < 1:
        logging.error("GEMINI_MAX_WORKERS debe ser mayor o igual que 1.")
        return 2

    if MAX_PAGES_PER_CHUNK < 1:
        logging.error(
            "GEMINI_MAX_PAGES_PER_CHUNK debe ser mayor o igual que 1."
        )
        return 2

    pdf_files = find_pdfs(SOURCE_DIR)

    if not pdf_files:
        logging.info(
            "No se encontraron archivos PDF en: %s",
            SOURCE_DIR,
        )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    worker_count = min(MAX_WORKERS, len(pdf_files))

    logging.info(
        "Se procesarán %d PDF con %d workers. Modelo: %s. "
        "Máximo inicial: %d páginas por fragmento.",
        len(pdf_files),
        worker_count,
        MODEL,
        MAX_PAGES_PER_CHUNK,
    )

    successes: list[Path] = []
    partials: list[tuple[Path, TranscriptionResult]] = []
    failures: list[tuple[Path, Exception]] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="gemini-pdf",
    ) as executor:
        future_to_pdf = {
            executor.submit(transcribe_with_retries, pdf_path): pdf_path
            for pdf_path in pdf_files
        }

        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]

            try:
                result = future.result()

                if result.omissions:
                    partials.append((pdf_path, result))
                else:
                    successes.append(result.output_path)
            except Exception as error:
                failures.append((pdf_path, error))
                logging.exception(
                    "Falló definitivamente la transcripción de %s",
                    pdf_path.name,
                )

    logging.info(
        "Proceso terminado. Correctos: %d. Incompletos: %d. Fallidos: %d.",
        len(successes),
        len(partials),
        len(failures),
    )

    if partials:
        print("\nArchivos incompletos guardados:", file=sys.stderr)

        for pdf_path, result in partials:
            print(
                f"  - {pdf_path.name} -> {result.output_path}",
                file=sys.stderr,
            )
            for omission in result.omissions:
                print(
                    f"      páginas {omission.start_page}-{omission.end_page}: "
                    f"{type(omission.error).__name__}: {omission.error}",
                    file=sys.stderr,
                )

    if failures:
        print("\nArchivos fallidos:", file=sys.stderr)

        for pdf_path, error in failures:
            print(
                f"  - {pdf_path.name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )

    return 1 if partials or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
