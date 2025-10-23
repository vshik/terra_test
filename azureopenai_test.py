import asyncio
from openai import AsyncAzureOpenAI

# --- Replace with your real details ---
AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE-NAME.openai.azure.com/"
AZURE_OPENAI_KEY = "YOUR-API-KEY"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # or gpt-4o / gpt-4.1
API_VERSION = "2025-03-01-preview"

async def test_azure_openai():
    client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=API_VERSION,
    )

    response = await client.responses.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello from Azure OpenAI async test!"}
        ],
    )

    print("Connected successfully!")
    print("Response:", response.output[0].content[0].text)

if __name__ == "__main__":
    asyncio.run(test_azure_openai())
