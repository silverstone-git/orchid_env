import gradio as gr
from daytona import Daytona, DaytonaConfig
from dotenv import load_dotenv
from os import getenv

# Load environment variables (e.g., DAYTONA_API_KEY)
load_dotenv()

def run_daytona_hello():
    """
    Creates a Daytona sandbox and runs a 'Hello World' python command,
    returning the result to the UI.
    """
    try:
        # Initialize Daytona with config from environment
        config = DaytonaConfig(
            api_key=getenv("DAYTONA_API_KEY", ""), 
        )
        daytona = Daytona(config)
        
        # Create a new sandbox
        sandbox = daytona.create()
        
        # Run the 'Hello World' command
        response = sandbox.process.code_run('print("Hello World!")')
        
        return response.result
    except Exception as e:
        return f"Error encountered: {str(e)}"

# Define the Gradio UI
with gr.Blocks(title="Daytona Hello World") as demo:
    gr.Markdown("# Daytona Hello World")
    gr.Markdown("Click the button below to execute a 'Hello World' script in a remote Daytona sandbox.")
    
    with gr.Column():
        run_btn = gr.Button("Run Hello World on Daytona", variant="primary")
        output_text = gr.Textbox(label="Sandbox Output", interactive=False, placeholder="Waiting for execution...")
    
    # Wire the button click event
    run_btn.click(
        fn=run_daytona_hello,
        inputs=None,
        outputs=output_text
    )

if __name__ == "__main__":
    # Launch the application
    demo.launch()
