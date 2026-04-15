"""Quick integration test to verify all components work together."""
from __future__ import annotations


def main():
    print("Testing CLI integration...")
    
    # Load config
    from src.config import load_config
    cfg = load_config("configs/app.yaml")
    print(f"[OK] Config loaded: LLM={cfg.llm.provider}, TTS={cfg.tts.provider}, STT={cfg.stt.provider if cfg.stt else 'None'}")
    
    # Build drivers
    from src.drivers import build_drivers
    llm, tts, stt = build_drivers(cfg)
    print(f"[OK] Drivers built: LLM={type(llm).__name__}, TTS={type(tts).__name__}, STT={type(stt).__name__}")
    
    # Test text normalization
    from src.text_normalizer import normalize_for_speech
    test = normalize_for_speech("December 6, 2025")
    print(f'[OK] Text normalization: "December 6, 2025" -> "{test}"')
    
    # Test memory extraction
    from src.memory import extract_fact_from_statement
    fact = extract_fact_from_statement("My birthday is March 15")
    print(f"[OK] Memory extraction: {fact}")
    
    # Test context awareness
    from src.context_awareness import TimeContext
    tc = TimeContext()
    print(f"[OK] Context: {tc.get_greeting()}, period={tc.get_period()}")
    
    # Test conversation memory
    from src.conversation import get_memory
    mem = get_memory()
    print(f"[OK] Conversation memory: max_turns={mem.max_turns}")
    
    # Test command handlers
    from src.commands import handle_weather_command
    handled, response = handle_weather_command("what is the weather in milwaukee")
    print(f"[OK] Weather command: handled={handled}")
    
    # Test user identity
    from src.user_identity import get_registry
    registry = get_registry()
    users = registry.list_users()
    print(f"[OK] User registry: {len(users)} users enrolled")
    
    # Test logging
    from src.logging_config import get_logger
    logger = get_logger("integration_test")
    logger.info("Logging works!")
    print("[OK] Logging system")
    
    print("")
    print("=== INTEGRATION TEST PASSED ===")


if __name__ == "__main__":
    main()

