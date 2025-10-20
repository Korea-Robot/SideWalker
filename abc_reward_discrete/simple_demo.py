"""
간단한 추론 데모 스크립트
"""
import torch
import numpy as np
from inference import MetaUrbanInference

def quick_demo():
    """빠른 데모 실행"""
    
    # 체크포인트 경로 (실제 경로로 수정하세요)
    actor_path = "metaurban_discrete_actor_multimodal_final.pt"
    
    try:
        # 모델 로드
        print("Loading model...")
        inference = MetaUrbanInference(
            actor_checkpoint_path=actor_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("✅ Model loaded successfully!")
        
        # 한 번의 에피소드 실행
        print("\n🚀 Running episode...")
        result = inference.run_episode(
            max_steps=200,
            deterministic=True,
            save_video=False
        )
        
        # 결과 출력
        print(f"\n📊 Episode Results:")
        print(f"   Total Reward: {result['total_reward']:.2f}")
        print(f"   Steps Taken: {result['steps']}")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Crashed: {'❌' if result['crash'] else '✅'}")
        
        # 행동 분포 분석
        if result['actions']:
            steering_actions = [a[0] for a in result['actions']]
            throttle_actions = [a[1] for a in result['actions']]
            
            print(f"\n🎮 Action Statistics:")
            print(f"   Steering range: {min(steering_actions)} ~ {max(steering_actions)}")
            print(f"   Throttle range: {min(throttle_actions)} ~ {max(throttle_actions)}")
            print(f"   Most used steering: {max(set(steering_actions), key=steering_actions.count)}")
            print(f"   Most used throttle: {max(set(throttle_actions), key=throttle_actions.count)}")
        
        inference.close()
        print("\n✨ Demo completed!")
        
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found: {actor_path}")
        print("Please make sure you have trained the model and saved the checkpoint.")
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_demo():
    """대화형 데모"""
    actor_path = "metaurban_discrete_actor_multimodal_final.pt"
    
    try:
        inference = MetaUrbanInference(actor_checkpoint_path=actor_path)
        print("✅ Model loaded! Starting interactive demo...")
        
        while True:
            print("\n" + "="*50)
            print("Interactive Demo Menu:")
            print("1. Run single episode")
            print("2. Run episode with video")
            print("3. Run evaluation (5 episodes)")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                result = inference.run_episode(deterministic=True)
                print(f"Reward: {result['total_reward']:.2f}, "
                      f"Success: {result['success']}")
                      
            elif choice == "2":
                result = inference.run_episode(
                    deterministic=True, 
                    save_video=True,
                    video_path="demo_video.mp4"
                )
                print(f"Video saved! Reward: {result['total_reward']:.2f}")
                
            elif choice == "3":
                results = inference.evaluate(num_episodes=5)
                print(f"Success rate: {results['success_rate']:.2%}")
                
            elif choice == "4":
                break
                
            else:
                print("Invalid choice!")
        
        inference.close()
        print("👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # 간단한 데모 실행
    quick_demo()
    
    # 대화형 데모를 원한다면 아래 주석을 해제하세요
    # interactive_demo()