from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wjs_mcp")

@mcp.tool()
def add(a: int, b: int) -> int:
    """add two numbers together"""
    return a + b

@mcp.tool()
def weather(city: str):
    """get someone city weather
    Args:
        city: city name
    """
    return "city" + city + "天气不错"


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Greet a person by name"""
    return f"Hello, {name}"

if __name__ == "__main__":
    # mcp.run(transport="streamable-http")
    mcp.run(transport="stdio")



