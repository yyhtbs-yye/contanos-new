import sys
import os
import asyncio

if __name__ == "__main__":
    # Save current working directory
    original_cwd = os.getcwd()

    # Get the project root (one level up from cmd directory)
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, project_root)

    # Get the config file path (in the cmd directory)
    config_path = os.path.join(os.path.dirname(__file__), 'dev_pose_estimation_config.yaml')

    # Switch to yolox directory
    yolox_dir = os.path.join(project_root, 'projects', 'prj-yolox-onnx')
    os.chdir(yolox_dir)

    # Add to Python path
    sys.path.insert(0, yolox_dir)

    try:
        # Modify sys.argv to include the config file path
        if '--config' not in sys.argv:
            sys.argv.extend(['--config', config_path])
        
        # Now we can import the YAML version
        from yolox_main_yaml import main as yolox_main
        
        import logging
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Run the main function
        asyncio.run(yolox_main())
        
    except Exception as e:
        logging.error(f"Error starting YOLO service: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)