import logging
import gradio as gr
from modules import scripts
from differential_diffusion.differential_diffusion import DifferentialDiffusion

class DifferentialDiffusionScript(scripts.Script):
    def __init__(self):
        self.enabled = False

    sorting_priority = 18

    def title(self):
        return "Differential Diffusion for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Enable or disable Differential Diffusion.</i></p>")
            enabled = gr.Checkbox(label="Enable Differential Diffusion", value=self.enabled)

            enabled.change(
                lambda x: self.update_enabled(x),
                inputs=[enabled]
            )

        return (enabled,)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 1:
            self.enabled = args[0]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        unet = DifferentialDiffusion().apply(unet)[0]
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update({
            "differential_diffusion_enabled": self.enabled,
        })

        logging.debug(f"Differential Diffusion: Enabled: {self.enabled}")

        return
