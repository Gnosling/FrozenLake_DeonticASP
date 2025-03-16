import os

from src import *
from src.bayesian_optimization import bayesian_optimization
from src.configs import configs

from PIL import Image

controller = Controller()

# experiments = [config for config in configs.keys() if config.startswith("E7") or config.startswith("E8") or config.startswith("E9")]
# for elem in experiments:
#     controller.plot_experiment(elem)


# experiments = ["B1_newstates", "C1_traverser", "C1_present", "C1_traverser_with_present"]
# controller.plot_compare_of_experiments(experiments, "3x3_B", True, False, 4)
#
# experiments = ["C2_normless", "C2_traverser", "C2_safe", "C2_traverser_with_safe"]
# controller.plot_compare_of_experiments(experiments, "4x4_B",True, False, 6)
#
# experiments = ["C3_rewards", "C3_scaling", "C3_violations", "C3_weakconstrains"]
# controller.plot_compare_of_experiments(experiments, "4x4_C", True, False, 4)
#
# experiments = ["B5_newstates", "C4_traverser", "C4_moving", "C4_strictly_moving", "C4_traverser_with_moving"]
# controller.plot_compare_of_experiments(experiments,"6x4_B",  True, False, 19)
#
# experiments = ["B6_newstates", "C5_safe", "C5_moving", "C5_safe_first", "C5_moving_first", "C5_equal"]
# controller.plot_compare_of_experiments(experiments, "7x4_A", True, False, 12)
#
# experiments = ["C6_traverser_first", "C6_safe_first", "C6_equal"]
# controller.plot_compare_of_experiments(experiments, "7x4_B", True, False, 15)
#
# experiments = ["B7_greedy", "C7_normless", "C7_moving", "C7_strictly_moving", "C7_traverser_with_moving"]
# controller.plot_compare_of_experiments(experiments, "7x4_C", True, False, 19)
#
# experiments = ["C8_normless", "C8_no_presents", "C8_presents", "C8_moving"]
# controller.plot_compare_of_experiments(experiments, "7x4_D", True, False, 20)
#
# experiments = ["B8_newstates", "C9_traverser", "C9_traverser_over_safe"]
# controller.plot_compare_of_experiments(experiments, "8x8_A", True, False, 13)
#
# experiments = ["C10_normless", "C10_safe", "C10_presents", "C10_safe_with_presents_1", "C10_safe_with_presents_2"]
# controller.plot_compare_of_experiments(experiments, "8x8_B", True, False, 18)
#
# experiments = ["D1_present", "D1_no_present", "D1_both"]
# controller.plot_compare_of_experiments(experiments, "3x3_B", True, False, 23)
#
# experiments = ["D2_normal", "D2_ross"]
# controller.plot_compare_of_experiments(experiments, "4x4_A", True, False, 1)
#
# experiments = ["D3_factual", "D3_deontic"]
# controller.plot_compare_of_experiments(experiments, "4x4_B", True, False, 1)
#
# experiments = ["D4_F(R)_F(M)", "D4_O(R)_F(M)", "D4_F(R)_O(M)", "D4_O(R)_O(M)"]
# controller.plot_compare_of_experiments(experiments, "7x4_A", True, False, 7)
#
# experiments = ["D5_newstates", "D5_full"]
# controller.plot_compare_of_experiments(experiments, "8x8_B", True, False, 5)
#

# experiments = ["E1_training_guard", "E1_training_fix", "E1_training_opt_shaping", "E1_training_full_shaping"]
# controller.plot_compare_of_experiments(experiments, "4x4_B", True, True, 2)
# #
# experiments = ["E2_post_guard", "E2_post_fix", "E2_post_opt_shaping", "E2_post_full_shaping"]
# controller.plot_compare_of_experiments(experiments, "4x4_B", True, True, 2)
#
# experiments = ["E3_guard", "E3_fix", "E3_opt_shaping", "E3_full_shaping", "E3_init_state_function", "E3_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "6x4_B", True, True, 2)
#
# experiments = ["E4_guard", "E4_fix", "E4_opt_shaping", "E4_full_shaping", "E4_init_state_function", "E4_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "7x4_B", True, True, 34)
# #
# experiments = ["E5_guard", "E5_fix", "E5_opt_shaping", "E5_full_shaping", "E5_init_state_function", "E5_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "7x4_B", True, True, 34)
# #
# experiments = ["E6_guard", "E6_fix", "E6_opt_shaping", "E6_full_shaping", "E6_init_state_function", "E6_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "7x4_B", True, True, 34)

# experiments = ["E7_guard", "E7_fix", "E7_full_shaping", "E7_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "8x8_A", True, True, 34)
#
# experiments = ["E8_guard", "E8_fix", "E8_full_shaping", "E8_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "8x8_A", True, True, 34)
#
# experiments = ["E9_guard", "E9_fix", "E9_full_shaping", "E9_init_action_penalty"]
# controller.plot_compare_of_experiments(experiments, "8x8_A", True, True, 34)
#
experiments = ["E10_guard", "E10_fix", "E10_full_shaping", "E10_init_action_penalty"]
controller.plot_compare_of_experiments(experiments, "8x8_A", True, True, 34)



def stack_images_vertically(image_files, spacing=10, output_file="stacked.png"):
    images = [Image.open(img) for img in image_files]
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
    combined = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))

    y_offset = 0
    for img in images:
        x_offset = (max_width - img.width) // 2
        combined.paste(img, (x_offset, y_offset), img if img.mode == "RGBA" else None)
        y_offset += img.height + spacing

    combined.save(output_file)
    print(f"Stacked image saved as {output_file}")


os.chdir(os.path.join("plots", "Tables"))

# image_files = ["Comparison_of_B1_newstates_C1_traverser_C1_present_C1_traverser_with_present.png",
#                "Comparison_of_C2_normless_C2_traverser_C2_safe_C2_traverser_with_safe.png",
#                "Comparison_of_C3_rewards_C3_scaling_C3_violations_C3_weakconstrains.png"]
# stack_images_vertically(image_files, spacing=0, output_file="Comparison_C1_C2_C3.png")
# image_files = ["Comparison_of_B5_newstates_C4_traverser_C4_moving_C4_strictly_moving_C4_traverser_with_moving.png",
#                "Comparison_of_B6_newstates_C5_safe_C5_moving_C5_safe_first_C5_moving_first_C5_equal.png",
#                "Comparison_of_C6_traverser_first_C6_safe_first_C6_equal.png",
#                "Comparison_of_B7_greedy_C7_normless_C7_moving_C7_strictly_moving_C7_traverser_with_moving.png"]
# stack_images_vertically(image_files, spacing=0, output_file="Comparison_C4_C5_C6_C7.png")
# image_files = ["Comparison_of_C8_normless_C8_no_presents_C8_presents_C8_moving.png",
#                "Comparison_of_B8_newstates_C9_traverser_C9_traverser_over_safe.png",
#                "Comparison_of_C10_normless_C10_safe_C10_presents_C10_safe_with_presents_1_C10_safe_with_presents_2.png"]
# stack_images_vertically(image_files, spacing=0, output_file="Comparison_C8_C9_C10.png")
# image_files = ["Comparison_of_D1_present_D1_no_present_D1_both.png",
#                "Comparison_of_D2_normal_D2_ross.png",
#                "Comparison_of_D3_factual_D3_deontic.png",
#                "Comparison_of_D4s.png",
#                "Comparison_of_D5_newstates_D5_full.png"]
# stack_images_vertically(image_files, spacing=0, output_file="Comparison_D1_D2_D3_D4_D5.png")

# image_files = ["Comparison_of_E1_training_guard_E1_training_fix_E1_training_full_shaping_E1_training_opt_shaping.png",
#                "Comparison_of_E2_post_guard_E2_post_fix_E2_post_full_shaping_E2_post_opt_shaping.png",
#                "Comparison_of_E3_guard_E3_fix_E3_opt_shaping_E3_full_shaping_E3_init_state_function_E3_init_action_penalty.png"]
# stack_images_vertically(image_files, spacing=0, output_file="Comparison_E1_E2_E3.png")
#
# image_files = ["Comparison_of_E4_guard_E4_fix_E4_opt_shaping_E4_full_shaping_E4_init_state_function_E4_init_action_penalty.png",
#                "Comparison_of_E5_guard_E5_fix_E5_opt_shaping_E5_full_shaping_E5_init_state_function_E5_init_action_penalty.png",
#                "Comparison_of_E6_guard_E6_fix_E6_opt_shaping_E6_full_shaping_E6_init_state_function_E6_init_action_penalty.png"]
# stack_images_vertically(image_files, spacing=0, output_file="Comparison_E4_E5_E6.png")

image_files = ["Comparison_of_E7_guard_E7_fix_E7_full_shaping_E7_init_action_penalty.png",
               "Comparison_of_E8_guard_E8_fix_E8_full_shaping_E8_init_action_penalty.png",
               "Comparison_of_E9_guard_E9_fix_E9_full_shaping_E9_init_action_penalty.png",
               "Comparison_of_E10_guard_E10_fix_E10_full_shaping_E10_init_action_penalty.png"]
stack_images_vertically(image_files, spacing=0, output_file="Comparison_E7_E8_E9_E10.png")