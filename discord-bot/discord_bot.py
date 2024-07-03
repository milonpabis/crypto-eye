import os
import discord


SECRET = os.environ.get("DISCORD_BTC_SECRET")
INTENTS = discord.Intents.default()
INTENTS.message_content = True

class StockTool(discord.Client):
    
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    async def on_message(self, message):
        print(f'{message.channel}: {message.author}: {message.author.name}: {message.content}')
        channel = message.channel
        if message.content.startswith("!predict"):
            await channel.send(f"predictions:")
            await channel.send(f":)")


client = StockTool(intents=INTENTS)
client.run(SECRET)

