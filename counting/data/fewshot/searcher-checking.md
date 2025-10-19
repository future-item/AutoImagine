You will receive an image of a 3D scene from a specific viewpoint, and the name of the object to be found. You need to determine whether the target object exists in the image.

Since the image is rendered by the edited 3D scene, please ignore artifacts like shadows, holes or floating spots. Some areas in the image are masked with black paint, you should ignore them as well.

Important Rule: In the image, all pure black area must be considered as illegal areas!

You should proceed step by step as follows:

1. Look for the target object in the image.
   * Search for objects that match the described target object.
   * Ignore any pure black areas as they are illegal and should not be considered.

2. Evaluate the found objects.
   * If you find objects that match the target description and are not completely black, consider them as valid.
   * If all matching objects are completely black, they should be ignored.

3. Make your decision.
   * If you find at least one valid (non-black) object that matches the target, output ['found'].
   * If you don't find any valid objects or all matching objects are completely black, output ['missing'].

### Example 1:
Input Image: <image>
Object to be found: French fries
Output: 
1. Search for the Target Object:
The task is to find French fries in the image.
Upon inspecting the image, I can identify several elongated, yellowish, fried potato slices that are consistent with the appearance of French fries.
These objects are clearly visible on the white paper background.
2. Evaluate the Found Objects:
Some of the objects in the image are located in pure black areas, which are explicitly marked as illegal areas and must be ignored.
However, there are at least three clearly visible French fries that are not in any black area, located on the right and lower parts of the image.
These fries are not covered or obscured, and they match the expected shape, color, and texture.
3. Decision:
Since there are valid, visible French fries that are not in the illegal (black) areas, the correct result is:
['found']

### Example 2:
Input Image: <image>
Object to be found: French fries
Output: 
1. Search for the Target Object:
The task is to identify French fries in the image.
French fries are typically golden yellow, elongated, and have a cooked potato appearance.
In this image, all visible objects that resemble the shape and size of French fries are entirely covered with pure black paint.
2. Evaluation of the Found Objects:
Every potential fry-shaped object is inside the illegal black areas.
According to the rules, black areas must be ignored, and any object entirely inside them is considered invalid.
No valid French fries are visible in any unmasked (legal) area of the image.
3. Decision:
Since no valid, visible French fries exist outside the black areas, the final result is:
['missing']

Now you need to refer to the previous examples to make a reasonable judgment on the following image:
Input Image: <image>
Object to be found: {object_name} 