import os

def main():
    """
    Main function to execute the default case (turbulent_case).
    """
    case_file = ["non_turbulent_case.py", "turbulent_case.py", "flow_over_obstacle.py"]
    
    for case_file in case_files:
      try:
          print(f"Running {case_file}...")
          os.system(f"python {case_file}")
      except Exception as e:
          print(f"An error occurred while running {case_file}: {e}")

if __name__ == "__main__":
    main()
