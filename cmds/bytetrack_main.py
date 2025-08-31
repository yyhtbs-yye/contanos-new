import sys
import os
import asyncio
import logging

if __name__ == "__main__":
    original_cwd = os.getcwd()
    # Get the project root (one level up from cmd directory)
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, project_root)

    # Get the config file path (in the cmd directory)
    config_path = os.path.join(os.path.dirname(__file__), 'dev_pose_estimation_config.yaml')

    bytetrack_dir = os.path.join(project_root, 'projects', 'prj-bytetrack-cpu')
    os.chdir(bytetrack_dir)

    sys.path.insert(0, bytetrack_dir)
    
    try:
        # Modify sys.argv to include the config file path
        if '--config' not in sys.argv:
            sys.argv.extend(['--config', config_path])
            
        from bytetrack_main_yaml import main as bytetrack_main
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)

        # Run the main function
        asyncio.run(bytetrack_main())
        
    except Exception as e:
        logging.error(f"Error starting ByteTrack service: {e}")
        raise
    finally:
        os.chdir(original_cwd)