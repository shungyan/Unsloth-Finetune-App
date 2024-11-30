import gradio as gr
import threading
import subprocess
import os
from tensorboard.program import TensorBoard

# Global variables to manage subprocesses
process = None
tb_process = None
stop_flag = threading.Event()  # To signal stopping the process

LOG_DIR = "./outputs"  # Set this to your TensorBoard log directory

# Function to start fine-tuning
def start_finetuning():
    global process, tb_process
    stop_flag.clear()  # Reset the stop flag

    # Start TensorBoard programmatically
    #tb_process = TensorBoard()
    #tb_process.configure(argv=[None, "--logdir", LOG_DIR, "--port", "6006"])
    #threading.Thread(target=tb_process.main, daemon=True).start()

    # Start the fine-tuning subprocess
    command = ["python", "finetune.py"]  # Replace with your fine-tuning script
    process = subprocess.Popen(command)

    # Return TensorBoard iframe to display the loss chart
    tensorboard_url = "http://localhost:6006"
    return f'<iframe src="{tensorboard_url}" width="100%" height="800px"></iframe>'

# Function to stop fine-tuning
def stop_finetuning():
    stop_flag.set()  # Signal to stop the process
    if process and process.poll() is None:  # Check if the process is running
        process.terminate()
    return "Fine-tuning stopped."

# Define Gradio components
with gr.Blocks() as app:
    gr.Markdown("# Fine-Tuning with TensorBoard")

    # Buttons to start/stop fine-tuning
    start_button = gr.Button("Start Fine-Tuning")
    stop_button = gr.Button("Stop Fine-Tuning")

    # Placeholder for TensorBoard iframe
    tensorboard_display = gr.HTML()

    # Connect event handlers to buttons
    start_button.click(
        fn=start_finetuning,
        inputs=[],
        outputs=[tensorboard_display]
    )
    stop_button.click(
        fn=stop_finetuning,
        inputs=[],
        outputs=[]
    )

app.launch()
