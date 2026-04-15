"""Voice enrollment script for adding new users to MOTHER.

Usage:
    python -m src.enroll_user                    # Interactive enrollment
    python -m src.enroll_user --list             # List enrolled users
    python -m src.enroll_user --delete <user_id> # Delete a user
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np

from mother.identity.speaker import (
    get_registry, UserRegistry, UserProfile,
    set_current_user, get_current_user
)


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    import sounddevice as sd
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                   channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def interactive_enrollment():
    """Interactive voice enrollment process."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice not installed. Run: pip install sounddevice")
        return 1
    
    registry = get_registry()
    
    print("\n" + "="*60)
    print("  MOTHER - Voice Enrollment")
    print("="*60)
    
    # Get user name
    print("\nPlease enter your name:")
    display_name = input("> ").strip()
    if not display_name:
        print("Name cannot be empty.")
        return 1
    
    # Check if user already exists
    user_id = display_name.lower().replace(" ", "_")
    existing = registry.get_user(user_id)
    if existing:
        print(f"\nUser '{display_name}' already exists.")
        print("1. Re-enroll voice")
        print("2. Cancel")
        choice = input("> ").strip()
        if choice != "1":
            print("Cancelled.")
            return 0
        profile = existing
    else:
        # Create new user
        profile = registry.create_user(display_name)
        print(f"\nCreated user profile: {profile.user_id}")
    
    # Voice enrollment
    print("\n" + "-"*60)
    print("Voice Enrollment")
    print("-"*60)
    print("""
I need to record your voice to recognize you in the future.
Please speak naturally for about 15 seconds. You can:
  - Count from 1 to 30
  - Talk about your day
  - Read something aloud
  - Just chat naturally

Press ENTER when ready to start recording...
""")
    input()
    
    # Record multiple samples
    samples = []
    sample_rate = 16000
    
    # First recording
    print("\n[Recording 1 of 2]")
    audio1 = record_audio(8.0, sample_rate)
    samples.append(audio1)
    print("✓ Captured")
    
    time.sleep(0.5)
    
    # Second recording
    print("\n[Recording 2 of 2]")
    print("Please continue speaking (different content is fine)...")
    time.sleep(1.0)
    audio2 = record_audio(7.0, sample_rate)
    samples.append(audio2)
    print("✓ Captured")
    
    # Enroll voice
    print("\nProcessing voice profile...")
    success = registry.enroll_voice(profile.user_id, samples, sample_rate)
    
    if success:
        print("\n" + "="*60)
        print(f"  ✓ Voice enrolled successfully for {display_name}!")
        print("="*60)
        print("\nMOTHER will now recognize your voice automatically.")
        
        # Ask for preferences
        print("\nWould you like to set any preferences?")
        print("1. Concise responses (default)")
        print("2. Detailed responses")
        print("3. Balanced")
        pref = input("> ").strip()
        
        if pref == "2":
            profile.preferences["response_style"] = "detailed"
        elif pref == "3":
            profile.preferences["response_style"] = "balanced"
        else:
            profile.preferences["response_style"] = "concise"
        
        profile.save()
        print(f"\nPreferences saved. Welcome aboard, {display_name}!")
        
    else:
        print("\n✗ Voice enrollment failed. Please try again.")
        return 1
    
    return 0


def list_users():
    """List all enrolled users."""
    registry = get_registry()
    users = registry.list_users()
    
    print("\n" + "="*60)
    print("  Enrolled Users")
    print("="*60)
    
    if not users:
        print("\nNo users enrolled yet.")
        print("Run: python -m src.enroll_user")
        return
    
    for user_id in users:
        profile = registry.get_user(user_id)
        if profile:
            voice_status = "✓ Voice enrolled" if profile.voice_enrolled else "✗ No voice"
            print(f"\n  {profile.display_name} ({user_id})")
            print(f"    {voice_status}")
            print(f"    Last seen: {profile.last_seen[:10]}")
            print(f"    Style: {profile.preferences.get('response_style', 'concise')}")


def delete_user(user_id: str):
    """Delete a user."""
    registry = get_registry()
    profile = registry.get_user(user_id)
    
    if not profile:
        print(f"User '{user_id}' not found.")
        return 1
    
    print(f"\nAre you sure you want to delete '{profile.display_name}'?")
    print("This will remove all their data including voice profile.")
    confirm = input("Type 'yes' to confirm: ").strip().lower()
    
    if confirm == "yes":
        registry.delete_user(user_id)
        print(f"✓ User '{profile.display_name}' deleted.")
        return 0
    else:
        print("Cancelled.")
        return 0


def test_recognition():
    """Test voice recognition against enrolled users."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice not installed.")
        return 1
    
    registry = get_registry()
    
    if not registry.list_users():
        print("No users enrolled. Run enrollment first.")
        return 1
    
    print("\n" + "="*60)
    print("  Voice Recognition Test")
    print("="*60)
    print("\nSpeak for 3 seconds and I'll try to identify you...")
    time.sleep(1.0)
    
    audio = record_audio(3.0, 16000)
    
    print("\nAnalyzing...")
    user_id, confidence = registry.identify_speaker(audio, 16000)
    
    if user_id:
        profile = registry.get_user(user_id)
        print(f"\n✓ Identified: {profile.display_name}")
        print(f"  Confidence: {confidence*100:.1f}%")
        
        if confidence < 0.8:
            print("  (Low confidence - might want to re-enroll)")
    else:
        print(f"\n✗ Could not identify speaker")
        print(f"  Best match confidence: {confidence*100:.1f}%")
        print("  Threshold is 75%")
    
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="MOTHER Voice Enrollment")
    parser.add_argument("--list", action="store_true", help="List enrolled users")
    parser.add_argument("--delete", type=str, help="Delete a user by ID")
    parser.add_argument("--test", action="store_true", help="Test voice recognition")
    
    args = parser.parse_args(argv)
    
    if args.list:
        list_users()
        return 0
    elif args.delete:
        return delete_user(args.delete)
    elif args.test:
        return test_recognition()
    else:
        return interactive_enrollment()


if __name__ == "__main__":
    sys.exit(main())

