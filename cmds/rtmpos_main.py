import sys
import os
import asyncio

if __name__ == "__main__":
    original_cwd = os.getcwd()
    # Get the project root (one level up from cmd directory)  
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, project_root)

    # Get the config file path (in the cmd directory)
    config_path = os.path.join(os.path.dirname(__file__), 'dev_pose_estimation_config.yaml')

    rtmpose_dir = os.path.join(project_root, 'projects', 'prj-rtmpose-onnx')
    os.chdir(rtmpose_dir)

    sys.path.insert(0, rtmpose_dir)
    
    try:
        # Modify sys.argv to include the config file path
        if '--config' not in sys.argv:
            sys.argv.extend(['--config', config_path])
            
        from rtmpose_main_yaml import main as rtmpose_main
        
        import logging
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Run the main function
        asyncio.run(rtmpose_main())

    except Exception as e:
        logging.error(f"Error starting RTMPose service: {e}")
        raise
    finally:
        os.chdir(original_cwd)