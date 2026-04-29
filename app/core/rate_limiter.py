from slowapi import Limiter
from slowapi.util import get_remote_address

# Key function: rate-limit per client IP.
# Swap get_remote_address for a user-id extractor when auth is added.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # No global default — limits are set per route
    headers_enabled=True,  # Inject RateLimit-* headers in responses
)
