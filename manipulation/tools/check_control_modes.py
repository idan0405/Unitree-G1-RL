"""Print the action/observation spaces and supported control modes of the
UnitreeG1PlaceAppleInBowl-v1 task. Handy when choosing a `control_mode`."""

import gymnasium as gym
import mani_skill.envs

def main():
    env = gym.make("UnitreeG1PlaceAppleInBowl-v1", obs_mode="state")
    print("Action space:", env.action_space)
    
    if hasattr(env.unwrapped, "supported_control_modes"):
        print("Supported control modes:", env.unwrapped.supported_control_modes)
    elif hasattr(env.unwrapped.agent, "supported_control_modes"):
        print("Agent supported control modes:", env.unwrapped.agent.supported_control_modes)
        
    print("Current control mode:", env.unwrapped.control_mode)
    print("Observation space:", env.observation_space)

if __name__ == "__main__":
    main()
