import asyncio
import os
import sys
import logging

if __name__ == "__main__":
    original_cwd = os.getcwd()
    project_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from cmd
    sys.path.insert(0, project_root)

    # Get the config file path (in the cmd directory)
    config_path = os.path.join(os.path.dirname(__file__), 'dev_pose_estimation_config.yaml')

    annotation_dir = os.path.join(project_root, 'projects', 'prj-annotation-cpu')

    os.chdir(annotation_dir)

    sys.path.insert(0, annotation_dir)
    
    try:
        # Modify sys.argv to include the config file path
        if '--config' not in sys.argv:
            sys.argv.extend(['--config', config_path])
            
        from annotation_main_yaml import main as annotation_main

        logging.basicConfig(level=logging.INFO)

        asyncio.run(annotation_main())
        
    except Exception as e:
        logging.error(f"Error starting annotation service: {e}")
        raise
    finally:
        os.chdir(original_cwd) 