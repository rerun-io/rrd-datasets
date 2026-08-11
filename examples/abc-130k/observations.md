# ABC-130k Observations

These findings come from surveying a subset of the source episodes, with a focus on details that aren’t covered in the [dataset card](https://huggingface.co/datasets/XDOF/ABC-130k).
We’re sharing them because they may be useful to others working with the dataset, especially when validating assumptions or building data pipelines around it.

The source revision is: [`29136bc`](https://huggingface.co/datasets/XDOF/ABC-130k/tree/29136bc9b9e38d320b00ffcddbbe4cd0e3278c58).
[Task and episode counts](#full-task-list) come from the split reports in that revision.
The surveyed subset includes about 450 episodes spanning all 197 tasks.

## Summary

The source revision includes 197 tasks and 130,703 episodes (train: 129,032, val: 1,671), totalling 3,590 hours.
Note that these counts differ from those reported in the [paper](https://abc.bot/abc.pdf) and on the [project site](https://abc.bot/).

The dataset is heterogeneous across several key dimensions.

| Dimension                 | Verdict      | Notes                                                                             |
| ------------------------- | ------------ | --------------------------------------------------------------------------------- |
| Duration                  | Diverse      | 5 s to 380 s, median 99 s                                                         |
| Frame rate                | Diverse      | synced or high-rate proprioception; cameras 18–60 Hz, proprioception up to 290 Hz |
| Camera model / resolution | Diverse      | 4 models, 6 resolutions                                                           |
| Camera layout             | Two variants | single or dual top camera                                                         |
| Annotation                | Two variants | none, or subtask labels                                                           |
| Gripper velocity/torque   | Two variants | on the gripper topic, or at index 6 of the arm state arrays                       |

## Example episodes

The table below shows representative converted segments, with one example for each notable case. The `Tag` column provides a short name for referring to each segment in the sections that follow.

| Tag                   | Segment id                                                                                        | Demonstrates                                                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `remove_the_shorts`   | `remove_the_shorts_from_the_hanger__0132ab84-8dd2-4fb7-ade9-12b68989720d`                         | single top camera · 640×480 D405 @30 Hz · no annotation · **short duration (17.4 s)**                                                                |
| `set_up_the_chess`    | `set_up_the_chess_pieces_on_the_board__001ecd72-c651-4389-a275-0fba7d0f438a`                      | single top camera · 640×480 OAK-1-W-97 @30 Hz · has annotation · **long duration (320.0 s)**                                                         |
| `fold_the_paper_box`  | `fold_the_paper_box__00444c81-7761-4e9c-a28f-6b1da1992eb2`                                        | **dual top camera** · 1920×1200 ZED_X H.265 @30 Hz (re-encoded to 640×480 H.264) · no annotation · **high-rate proprioception**                      |
| `screw_on_the_bottle` | `screw_on_the_bottle_caps__011d8c5b-b147-4293-b05c-a2567f939e66`                                  | single top camera · 640×480 D405 @59 Hz · **high-rate proprioception** · **the only `/barcode_scanner-barcode-scan` episode** (not included in .rrd) |
| `clip_the_underwear`  | `clip_the_underwear_to_the_hanger__0893975b-d8a5-4e92-9291-1f37c416b25a`                          | single top camera · **848×480** D405 · has annotation · **declares 60 fps, logs 30 Hz**                                                              |
| `fold_the_napkin`     | `fold_the_napkin_into_a_case_and_place_the_utensils_inside__00cb478a-aa77-4e5f-ac5a-e53f62914d3f` | single top camera · **1280×720** D405 @30 Hz · has annotation                                                                                        |
| `build_the_wood`      | `build_the_wood_block_tower__00784ada-77e4-4fdc-b26a-d76a1f24cab0`                                | single top camera · 640×480 D405 @29 Hz · no annotation · **missing `/left-wrist-camera-info`**                                                      |
| `remove_the_keys`     | `remove_the_keys_from_the_keyring__a8a5e956-840e-45e6-a949-02b211218877`                          | single top camera · **800×600** decxin @30 Hz · **high-rate proprioception** · **coded video is 800×608** · **identity intrinsics**                  |
| `place_the_utensils`  | `place_the_utensils_on_the_paper_napkin_and_roll_it_up__011640fb-8b08-4100-81b3-6e10f44708d3`     | single top camera · **1280×1024** decxin @30 Hz · **high-rate proprioception** · **identity intrinsics**                                             |

## Frame rate — synced or high-rate proprioception

Episodes fall into two timing layouts.

- **Synced**: every streaming topic shares one rate, matching the camera.
  Example: `remove_the_shorts`, where the cameras and all arm and gripper topics log at 29.8 Hz.
- **High-rate proprioception**: cameras stay at 30–60 Hz while arm and gripper topics log at their native rate — commands near 200 Hz and states between 210 Hz and 290 Hz.
  Examples: `fold_the_paper_box` (cameras 30 Hz, commands 200 Hz, states 264–277 Hz) and `screw_on_the_bottle` (cameras 59 Hz, commands 200 Hz, states 233–244 Hz).

Other noticeable things:

- Every ZED_X and decxin episode measured high-rate, every OAK-1-W-97 episode measured synced, and the D405 episodes split between the two.
- Several 640×480 and 848×480 episodes declare 60 fps and log 30 Hz, and one episode declaring 30 fps logs 18.6 Hz.

## Gripper velocity and torque — arm array or gripper topic

The source writes a gripper's velocity and torque in one of two places, depending on the episode; the converted recordings unify them.

- **In the arm arrays**: `/<side>-arm-state` logs width-7 `velocity` and `torque` arrays with the gripper at index 6, while the gripper topic leaves those fields null.
- **On the gripper topic**: the arm arrays are width 6, and `/<side>-ee-state` carries the gripper's velocity and torque itself.

Gripper position always comes from the gripper topic, and the arm `position` array holds only the 6 joints in both layouts.
The converter detects the layout from the arm array width: converted arm entities hold only the 6 joints, and the gripper's signals always land under `/<side>/gripper/`.

## Cameras

Four camera models and six resolutions appear in the original source, though the dataset documentation mentions only two models ("Realsense" and "ZED_X").
`640×480 RealSense @30 Hz` is the dominant case.

TODO: convert the following block into a table.

```text
640×480    Intel RealSense D405   declares 30 or 60 fps
848×480    Intel RealSense D405
1280×720   Intel RealSense D405
1920×1200  ZED_X                  dual top camera, H.265.  # (re-encoded to 640×480 H.264 in .rrd)
640×480    OAK-1-W-97
800×600    decxin                 coded video is 8 px taller
1280×1024  decxin
```

Wrist cameras report the same model as the top camera, except on ZED stations, where they report `Zed X One GS`.
The `station` property carries `RealSense` or `ZED-X` for those two families and the raw model string otherwise (`oak-1-w-97`, `decxin`).

The ZED episodes carry the highest resolution and are the only H.265 sources; everything else is H.264.
The conversion downscales them for the re-encoded viewing layer and rescales the `Pinhole` to match.

## Annotations

About one third carry subtask labels in a sibling `annotation.mcap`.

## Edge cases

- **Missing wrist calibration**: There are episodes that carry left wrist video with no `/left-wrist-camera-info`. (example: `build_the_wood`).
- **Video resolution disagrees with `camera-info`**: the decxin 800×600 episodes decode to 800×608 while camera info and episode metadata say 800×600 (example: `remove_the_keys`).
  The source H.264 stream is coded at 800×608 and sets no cropping window.
- **Identity intrinsics**: every decxin episode reports `fx = fy = 1` and `cx = cy = 0` (example: `place_the_utensils`).
- **Stray topic**: `/barcode_scanner-barcode-scan` appears in some episodes, logging at about 99 Hz (`screw_on_the_bottle`). This is not added to .rrd.

## Full task list

<details>
<summary>197 tasks with episode counts</summary>

| Task                                                                           | Train  | Val | Total  |
| ------------------------------------------------------------------------------ | ------ | --- | ------ |
| `arrange_the_flowers_into_the_vase`                                            | 203    | 4   | 207    |
| `assemble_a_carrot_with_lego`                                                  | 36     | —   | 36     |
| `attach_the_microphones_to_the_stand`                                          | 235    | 3   | 238    |
| `build_the_wood_block_tower`                                                   | 599    | 9   | 608    |
| `clean_the_litter_box`                                                         | 172    | 5   | 177    |
| `clear_the_kitchen_counter`                                                    | 90     | 1   | 91     |
| `clip_the_socks_to_the_hanger`                                                 | 457    | 7   | 464    |
| `clip_the_underwear_to_the_hanger`                                             | 151    | 3   | 154    |
| `connect_and_route_the_hose`                                                   | 75     | 2   | 77     |
| `decorate_the_small_christmas_tree`                                            | 180    | 5   | 185    |
| `distribute_texas_hold_em_gaming_equipment`                                    | 421    | 7   | 428    |
| `dress_the_teddy_bear`                                                         | 70     | 1   | 71     |
| `erase_the_whiteboard`                                                         | 1,276  | 17  | 1,293  |
| `fill_the_litter_box_with_clean_litter`                                        | 216    | 4   | 220    |
| `fold_a_paper_plane_with_a_square_sheet_of_paper_fold_left_side_to_right_side` | 591    | 7   | 598    |
| `fold_a_paper_plane_with_a_square_sheet_of_paper_pick_it_up_from_the_middle`   | 1,765  | 8   | 1,773  |
| `fold_and_stack_the_long_sleeve_shirts`                                        | 631    | 1   | 632    |
| `fold_and_stack_the_mixed_laundry_pile`                                        | 278    | 3   | 281    |
| `fold_and_stack_the_shorts`                                                    | 753    | 3   | 756    |
| `fold_and_stack_the_skirts`                                                    | 1,181  | 9   | 1,190  |
| `fold_and_stack_the_t_shirts`                                                  | 11,009 | 82  | 11,091 |
| `fold_and_stack_the_tank_tops`                                                 | 703    | 1   | 704    |
| `fold_and_stack_the_towels`                                                    | 384    | 5   | 389    |
| `fold_and_stack_the_trousers`                                                  | 607    | 5   | 612    |
| `fold_the_inside_out_t_shirt`                                                  | 527    | 11  | 538    |
| `fold_the_napkin_into_a_case_and_place_the_utensils_inside`                    | 470    | 4   | 474    |
| `fold_the_napkin_place_the_utensils_inside_and_roll_it_up`                     | 399    | 2   | 401    |
| `fold_the_paper_box`                                                           | 2,387  | 30  | 2,417  |
| `insert_the_pens_into_the_pen_caps`                                            | 784    | 16  | 800    |
| `insert_the_plug`                                                              | 808    | 7   | 815    |
| `insert_the_plug_into_the_switch_port`                                         | 427    | 4   | 431    |
| `insert_the_wireless_bluetooth_earbuds_into_the_charging_case`                 | 2,095  | 17  | 2,112  |
| `install_the_large_spring`                                                     | 114    | —   | 114    |
| `install_the_water_faucets`                                                    | 352    | 7   | 359    |
| `load_the_batteries_into_the_remote_control`                                   | 1,167  | 7   | 1,174  |
| `load_the_bowls_into_the_dish_rack`                                            | 601    | 11  | 612    |
| `load_the_cups_into_the_dish_rack`                                             | 477    | 9   | 486    |
| `load_the_mixed_dishes_into_the_dish_rack`                                     | 568    | 9   | 577    |
| `load_the_plates_into_the_dish_rack`                                           | 1,335  | 23  | 1,358  |
| `load_the_staples_into_the_stapler`                                            | 547    | 5   | 552    |
| `lock_with_the_key`                                                            | 636    | 11  | 647    |
| `open_and_lay_out_the_surgical_kit`                                            | 307    | 3   | 310    |
| `open_take_out_and_arrange_the_luggage`                                        | 182    | 4   | 186    |
| `open_the_pen_caps`                                                            | 516    | 7   | 523    |
| `open_the_umbrella`                                                            | 106    | 2   | 108    |
| `open_the_zip_top_bag_and_remove_the_fake_food`                                | 763    | 21  | 784    |
| `organize_the_chemistry_lab_equipment`                                         | 485    | 12  | 497    |
| `organize_the_condiment_bottles`                                               | 1,442  | 25  | 1,467  |
| `organize_the_desk`                                                            | 573    | 8   | 581    |
| `organize_the_hangers`                                                         | 115    | 1   | 116    |
| `organize_the_makeup`                                                          | 731    | 9   | 740    |
| `organize_the_medicine_kit`                                                    | 267    | 6   | 273    |
| `organize_the_mixed_jewelry`                                                   | 168    | 3   | 171    |
| `organize_the_sunglasses`                                                      | 434    | 6   | 440    |
| `pack_the_candies`                                                             | 382    | 6   | 388    |
| `pack_the_chocolate`                                                           | 539    | 5   | 544    |
| `pack_the_cookies`                                                             | 464    | 8   | 472    |
| `pack_the_luggage`                                                             | 570    | 10  | 580    |
| `pack_the_student_bag`                                                         | 1,978  | 20  | 1,998  |
| `pack_the_takeout_coffee`                                                      | 1,864  | 12  | 1,876  |
| `pack_the_tool_kit`                                                            | 337    | 2   | 339    |
| `pack_up_texas_hold_em_set`                                                    | 407    | 7   | 414    |
| `pack_up_the_badminton_gear`                                                   | 286    | 8   | 294    |
| `pack_up_the_instrument_practice_props`                                        | 263    | 4   | 267    |
| `paint_the_nails`                                                              | 305    | 1   | 306    |
| `pin_the_brooch_onto_the_clothing`                                             | 263    | 1   | 264    |
| `place_and_organize_the_beverage_onto_the_shelf`                               | 222    | 5   | 227    |
| `place_and_organize_the_candy_bags_onto_the_shelf`                             | 142    | —   | 142    |
| `place_and_organize_the_canned_foods_onto_the_counter`                         | 200    | 6   | 206    |
| `place_and_organize_the_chips_bags_onto_the_shelf`                             | 84     | 5   | 89     |
| `place_and_organize_the_cleaning_sponges_onto_the_shelf`                       | 194    | 4   | 198    |
| `place_and_organize_the_fake_fruits_in_the_fruit_bowl`                         | 815    | 13  | 828    |
| `place_and_organize_the_paper_towels_onto_the_shelf`                           | 201    | 4   | 205    |
| `place_and_organize_the_pasta_boxes_and_bags_onto_the_shelf`                   | 434    | 5   | 439    |
| `place_and_organize_the_plastic_toys_onto_the_shelf`                           | 300    | 2   | 302    |
| `place_and_organize_the_shaving_razors_onto_the_shelf`                         | 126    | 3   | 129    |
| `place_and_organize_the_shoes_in_the_shoe_cabinet`                             | 89     | 6   | 95     |
| `place_and_organize_the_shoes_onto_the_shoe_shelf`                             | 131    | 4   | 135    |
| `place_and_organize_the_stuffed_toys_onto_the_shelf`                           | 193    | 4   | 197    |
| `place_and_organize_the_toothpastes_onto_the_shelf`                            | 70     | 2   | 72     |
| `place_the_beverage_into_the_canvas_bag`                                       | 182    | 1   | 183    |
| `place_the_bra_in_the_laundry_bag_and_zip_it_closed`                           | 250    | 3   | 253    |
| `place_the_bread`                                                              | 1,362  | 27  | 1,389  |
| `place_the_coffee_filter_in_the_dripper`                                       | 529    | 16  | 545    |
| `place_the_cup_by_the_coaster`                                                 | 657    | 10  | 667    |
| `place_the_fake_bread_into_the_paper_bag`                                      | 293    | 8   | 301    |
| `place_the_fake_fruits_into_the_plastic_bag_and_tie_a_complex_knot`            | 238    | 7   | 245    |
| `place_the_flowers_into_the_vase`                                              | 493    | 9   | 502    |
| `place_the_food_in_the_zip_top_bag_and_seal_the_zipper`                        | 2,692  | 18  | 2,710  |
| `place_the_food_into_the_grocery_bag_and_tie_it`                               | 136    | 1   | 137    |
| `place_the_fruits_into_the_plastic_bag`                                        | 313    | 2   | 315    |
| `place_the_glasses_into_the_tray`                                              | 250    | 4   | 254    |
| `place_the_letter_into_the_envelope_and_seal_it`                               | 372    | 4   | 376    |
| `place_the_mixed_dishes_into_the_plastic_bin`                                  | 440    | 10  | 450    |
| `place_the_mixed_food_into_the_grocery_bags_and_tie_them`                      | 134    | 1   | 135    |
| `place_the_personal_care_products_into_the_canvas_bag`                         | 225    | 6   | 231    |
| `place_the_phone_on_the_phone_stand`                                           | 322    | 4   | 326    |
| `place_the_plates_into_the_plastic_bin_on_the_countertop`                      | 1,146  | 14  | 1,160  |
| `place_the_shirt_on_the_hanger`                                                | 247    | 3   | 250    |
| `place_the_shorts_on_the_hanger`                                               | 298    | 6   | 304    |
| `place_the_skirt_on_the_hanger`                                                | 217    | 3   | 220    |
| `place_the_snacks_into_the_paper_bag`                                          | 723    | 7   | 730    |
| `place_the_t_shirt_on_the_hanger`                                              | 750    | 19  | 769    |
| `place_the_tank_top_on_the_hanger`                                             | 341    | 7   | 348    |
| `place_the_trousers_on_the_hanger`                                             | 482    | 4   | 486    |
| `place_the_utensils_on_the_paper_napkin_and_roll_it_up`                        | 283    | 7   | 290    |
| `place_the_wine_glass_upside_down_on_the_wine_glass_rack`                      | 440    | 6   | 446    |
| `prepare_the_surgical_pack`                                                    | 129    | 1   | 130    |
| `prepare_toiletry_sets_of_a_toothbrush_and_toothpaste_on_the_tray`             | 200    | 5   | 205    |
| `pull_the_plug_from_the_switch_port`                                           | 327    | 4   | 331    |
| `pull_the_plug_off_the_socket`                                                 | 1,007  | 17  | 1,024  |
| `put_away_the_umbrella`                                                        | 216    | 5   | 221    |
| `put_the_credit_cards_into_the_card_holder`                                    | 2,574  | 28  | 2,602  |
| `put_the_files_into_the_folder`                                                | 513    | 11  | 524    |
| `put_the_keys_on_the_keyring`                                                  | 2,805  | 35  | 2,840  |
| `put_the_phone_into_the_phone_case`                                            | 584    | 14  | 598    |
| `put_the_photo_into_the_frame`                                                 | 898    | 7   | 905    |
| `put_the_pillow_into_the_pillowcase`                                           | 538    | 10  | 548    |
| `put_the_plastic_bottles_in_the_bin`                                           | 3,793  | 54  | 3,847  |
| `put_the_screwdriver_in_the_bin`                                               | 2,234  | 45  | 2,279  |
| `put_the_trash_bags_into_the_trash_bin`                                        | 521    | 7   | 528    |
| `put_the_wine_bottle_on_the_wine_rack`                                         | 214    | 10  | 224    |
| `remove_the_brooch_from_the_garment`                                           | 341    | 6   | 347    |
| `remove_the_keys_from_the_keyring`                                             | 745    | 12  | 757    |
| `remove_the_microphone_from_the_stand`                                         | 342    | 5   | 347    |
| `remove_the_phone_from_the_phone_stand`                                        | 481    | 11  | 492    |
| `remove_the_pillowcase_from_the_pillow`                                        | 664    | 12  | 676    |
| `remove_the_shirt_from_the_hanger`                                             | 212    | 3   | 215    |
| `remove_the_shorts_from_the_hanger`                                            | 384    | 6   | 390    |
| `remove_the_skirt_from_the_hanger`                                             | 297    | 4   | 301    |
| `remove_the_socks_clipped_to_the_hanger`                                       | 345    | 12  | 357    |
| `remove_the_t_shirt_from_the_hanger`                                           | 394    | 7   | 401    |
| `remove_the_tank_top_from_the_hanger`                                          | 483    | 7   | 490    |
| `remove_the_trousers_from_the_hanger`                                          | 439    | 8   | 447    |
| `remove_the_underwear_clipped_to_the_hanger`                                   | 136    | 4   | 140    |
| `roll_out_the_dumpling_wrappers_from_the_dough`                                | 133    | 1   | 134    |
| `roll_the_socks`                                                               | 586    | 3   | 589    |
| `roll_the_t_shirts`                                                            | 271    | 2   | 273    |
| `roll_the_ties`                                                                | 548    | 3   | 551    |
| `roll_the_towels`                                                              | 585    | 3   | 588    |
| `roll_the_underwear`                                                           | 378    | —   | 378    |
| `roll_up_the_long_pants`                                                       | 107    | —   | 107    |
| `screw_on_the_bottle_caps`                                                     | 370    | 5   | 375    |
| `serve_the_afternoon_tea`                                                      | 229    | 5   | 234    |
| `serve_the_lunch_box`                                                          | 407    | 6   | 413    |
| `serve_the_pet_food`                                                           | 229    | 4   | 233    |
| `set_the_dinner_table_for_one_person`                                          | 1,613  | 11  | 1,624  |
| `set_up_the_chess_pieces_on_the_board`                                         | 1,188  | 6   | 1,194  |
| `set_up_the_instrument_practice_props`                                         | 204    | 3   | 207    |
| `set_up_the_pet_bed`                                                           | 110    | —   | 110    |
| `set_up_the_pet_tent`                                                          | 98     | 1   | 99     |
| `snap_on_the_fuse_box`                                                         | 75     | —   | 75     |
| `sort_the_eating_utensils_into_containers`                                     | 536    | 5   | 541    |
| `sort_the_hair_cutting_tools`                                                  | 154    | 4   | 158    |
| `sort_the_legos_into_containers_by_color`                                      | 4,458  | 70  | 4,528  |
| `sort_the_pills_into_containers`                                               | 510    | 11  | 521    |
| `sort_the_screws_and_nuts_into_containers`                                     | 150    | 6   | 156    |
| `sort_the_stationery_into_containers`                                          | 2,079  | 11  | 2,090  |
| `sort_the_tools_into_containers`                                               | 609    | 7   | 616    |
| `stack_the_books_and_files`                                                    | 517    | 6   | 523    |
| `stack_the_hats`                                                               | 293    | 7   | 300    |
| `sweep_away_the_paper_scraps_from_the_table`                                   | 483    | 6   | 489    |
| `take_the_beverage_out_of_the_canvas_bag`                                      | 201    | 4   | 205    |
| `take_the_credit_cards_out_of_the_card_holder`                                 | 2,732  | 29  | 2,761  |
| `take_the_fake_bread_out_of_the_paper_bag`                                     | 253    | 1   | 254    |
| `take_the_fake_fruits_out_of_the_plastic_bag`                                  | 853    | 13  | 866    |
| `take_the_personal_care_products_out_of_the_canvas_bag`                        | 213    | 1   | 214    |
| `take_the_phone_out_of_the_phone_case`                                         | 734    | 8   | 742    |
| `take_the_photo_out_of_the_frame`                                              | 257    | 3   | 260    |
| `take_the_snacks_out_of_the_paper_bag`                                         | 643    | 7   | 650    |
| `take_the_wine_bottle_off_the_wine_rack`                                       | 211    | 3   | 214    |
| `take_the_wine_glass_off_the_wine_glass_rack`                                  | 780    | 18  | 798    |
| `throw_the_plastic_bottles_in_the_bin`                                         | 433    | 4   | 437    |
| `tie_the_bouquet_of_fake_flowers`                                              | 408    | 1   | 409    |
| `tie_the_shoes`                                                                | 449    | 7   | 456    |
| `turn_the_large_container_upside_down`                                         | 334    | 7   | 341    |
| `turn_the_mug_right_side_up`                                                   | 660    | 13  | 673    |
| `unload_the_batteries_from_the_remote_control`                                 | 407    | 7   | 414    |
| `unload_the_bowls_from_the_dish_rack`                                          | 320    | 8   | 328    |
| `unload_the_cups_from_the_dish_rack`                                           | 402    | 10  | 412    |
| `unload_the_mixed_dishes_from_the_dish_rack`                                   | 505    | 9   | 514    |
| `unload_the_plates_from_the_dish_rack`                                         | 1,333  | 8   | 1,341  |
| `unload_the_staples_from_the_stapler`                                          | 344    | 7   | 351    |
| `unlock_the_padlock_with_the_key`                                              | 1,844  | 30  | 1,874  |
| `unlock_with_the_key`                                                          | 627    | 12  | 639    |
| `unpack_the_express_box`                                                       | 260    | 5   | 265    |
| `unscrew_the_bottle_caps`                                                      | 294    | 5   | 299    |
| `untangle_the_cables`                                                          | 2,950  | 22  | 2,972  |
| `untie_the_knot_of_the_grocery_bags_and_take_out_the_mixed_food`               | 73     | 2   | 75     |
| `untie_the_knot_of_the_plastic_bag_and_take_out_the_fruit`                     | 329    | 7   | 336    |
| `unzip_the_laundry_bag_and_remove_the_bra`                                     | 770    | 12  | 782    |
| `wrap_the_cables`                                                              | 404    | 8   | 412    |
| `wrap_the_gift_box_in_paper`                                                   | 105    | 4   | 109    |
| `wrap_the_headphones`                                                          | 1,304  | 14  | 1,318  |
| `write_a_sentence_on_the_whiteboard`                                           | 1,394  | 7   | 1,401  |
| `zip_tie_the_wires`                                                            | 57     | —   | 57     |
| `zip_up_the_jacket`                                                            | 873    | 13  | 886    |

</details>
