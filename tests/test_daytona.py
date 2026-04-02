from daytona import Daytona, DaytonaConfig
from dotenv import load_dotenv
from os import getenv

load_dotenv()

# config = DaytonaConfig(api_key="YOUR_API_KEY")
config = DaytonaConfig(
    api_key=getenv("DAYTONA_API_KEY", ""), 
    )
daytona = Daytona(config)
sandbox = daytona.create()
response = sandbox.process.code_run('print("Hello World!")')
print(response.result)
