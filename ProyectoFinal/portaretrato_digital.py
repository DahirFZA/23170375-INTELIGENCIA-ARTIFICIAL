import gradio as gr
import torch
from transformers import pipeline
from diffusers import AutoPipelineForText2Image


DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
TIPO_DATO = torch.float16 if DISPOSITIVO == "cuda" else torch.float32

# Modelo 1: Reconocimiento automático de voz (Audio -> Texto)
modelo_voz = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=0 if DISPOSITIVO == "cuda" else -1,
)

# Modelo 2: Síntesis de imagen condicionada por texto (Texto -> Imagen)
modelo_imagen = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=TIPO_DATO,
    variant="fp16" if DISPOSITIVO == "cuda" else None,
).to(DISPOSITIVO)


def generar_retrato_desde_voz(ruta_audio, prompt_manual, estilo, historial):
    """Transcribe el audio (si existe), combina con el estilo y genera la imagen para el marco."""
    descripcion = ""

    if ruta_audio is not None:
        salida_whisper = modelo_voz(ruta_audio, generate_kwargs={"language": "spanish"})
        descripcion = salida_whisper["text"].strip()
    elif prompt_manual and prompt_manual.strip():
        descripcion = prompt_manual.strip()
    else:
        return None, "Graba un audio o escribe una descripción para generar el retrato.", historial

    prompt_enriquecido = (
        f"{descripcion}, {estilo}, high detail, sharp focus, aesthetic lighting, masterpiece"
    )

    # sd-turbo funciona de forma óptima con 1 a 2 pasos y guidance_scale=0.0
    pasos = 2 if DISPOSITIVO == "cuda" else 1
    imagen_generada = modelo_imagen(
        prompt=prompt_enriquecido,
        num_inference_steps=pasos,
        guidance_scale=0.0,
        width=512,
        height=512,
    ).images[0]

    nuevo_historial = historial + [(imagen_generada, f"{descripcion} ({estilo})")]
    return imagen_generada, descripcion, nuevo_historial


def limpiar_galeria():
    return None, "", [], []


with gr.Blocks(theme=gr.themes.Base(), title="Portaretrato Digital IA") as interfaz:
    gr.Markdown("# Portaretrato Digital con IA Generativa")
    gr.Markdown(
        "Describe por voz lo que deseas visualizar. El sistema transcribe tu voz con **Whisper** "
        "y sintetiza la obra con **SD-Turbo** para exhibirla en el marco digital."
    )

    estado_galeria = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            entrada_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="1. Habla al micrófono o sube un audio",
            )
            entrada_texto_respaldo = gr.Textbox(
                label="O escribe una descripción manual (opcional)",
                placeholder="Ej. Un astronauta descansando en una playa de Sinaloa al atardecer",
            )
            selector_estilo = gr.Dropdown(
                choices=[
                    "photorealistic photography",
                    "oil painting on canvas",
                    "digital concept art",
                    "watercolor illustration",
                    "cyberpunk neon style",
                    "studio ghibli anime style",
                ],
                value="photorealistic photography",
                label="2. Estilo visual del portaretrato",
            )
            with gr.Row():
                boton_crear = gr.Button("Proyectar en Portaretrato", variant="primary")
                boton_limpiar = gr.Button("Limpiar Marco")

            texto_transcrito = gr.Textbox(
                label="Texto detectado (Modelo 1: Whisper)",
                interactive=False,
            )

        with gr.Column(scale=2):
            marco_principal = gr.Image(
                label="Marco Digital Activo (Modelo 2: SD-Turbo)",
                height=480,
            )

    galeria_retratos = gr.Gallery(
        label="Colección guardada en el Portaretrato",
        columns=4,
        height=210,
        object_fit="cover",
    )

    boton_crear.click(
        fn=generar_retrato_desde_voz,
        inputs=[entrada_audio, entrada_texto_respaldo, selector_estilo, estado_galeria],
        outputs=[marco_principal, texto_transcrito, galeria_retratos],
    ).then(
        fn=lambda g: g,
        inputs=[galeria_retratos],
        outputs=[estado_galeria],
    )

    boton_limpiar.click(
        fn=limpiar_galeria,
        inputs=[],
        outputs=[marco_principal, texto_transcrito, galeria_retratos, estado_galeria],
    )


if __name__ == "__main__":
    interfaz.launch()