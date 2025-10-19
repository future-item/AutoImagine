You will receive an image that has been masked using SAM, where the black areas represent the masked-out parts. These black areas must be considered as background.
Your task is to verify if the not black area shown in the image matches the described object. If there are any minor artifacts, render noise, or small visual imperfections, ignore them unless they clearly resemble separate, distinct objects.
You should proceed step by step as follows:
1. Determine whether there is a single and complete object in the image.
   The object must be complete. If the object is incomplete or fragmented, it cannot be considered as the target object.
   If there is a single, complete object, check if it matches the target object. If it is, return ['yes'].
   To identify a single object, check if the non-black area in the image represents only one complete object, while the black area is clearly the surrounding background.
2. The object in the image might be blurry or incomplete.
   Ignore minor artifacts and focus on recognizing the object based on its overall color and shape.
3. Remember:
   All black areas in the image must be considered as background, even if these black areas represent part of the object itself. Black areas should not be considered part of the object you are trying to recognize.
4. Sometimes we may mistakenly segment containers that hold things, such as billiard tables or paper towels used to place food. 
   A notable feature of such objects is that they have two or more disconnected black backgrounds. 
   When you find that the object is a container, you should reply ['no'].

Step-by-Step Evaluation:
Step 1: Analyze if the image contains only one object and if it is complete.
Step 2: Evaluate the object despite any blurriness or imperfections.
Step 3: Analyze whether the image show a mistakenly segment containers that hold things.
Step 4: Based on the above analysis, output the result in the format in last line: ['yes'] or ['no'].

### Example 1:
Input Image: <image>
Object Description: French fries.
Output: 
### Reasoning:

1. Multiple Disconnected Black Regions:
   The image contains multiple black cutouts on a white surface. These black areas are not part of a single continuous object, indicating that the visible structure is likely a container or surface.

2. Nature of the Object:
   The visible object appears to be a paper towel or napkin used to hold or support multiple items (likely French fries). The segmented black areas are holes or cutouts where the fries would have been, not the fries themselves.

3. Violation of Background Rule:
   According to the rules, any black area is treated as background. In this image, since the black areas intersect with what would have been multiple fries, it fragments those fries, making none of them a complete object.

4. Conclusion – It's a Container:
   The white object with multiple cutouts and surrounding black background functions as a holder, not as the object of interest (French fries).

Therefore, this does not show a single, complete French fry — it shows a container or background surface, which disqualifies it.
['no']

### Example 2:
Input Image: <image>
Object Description: pills.
Output: 
### Reasoning:
1. Multiple Objects Present:
   The image shows several small, round yellow objects, each clearly separated by black background. This violates the requirement that the image must contain a single, complete object.

2. Not a Single Complete Object:
   Since the non-black areas represent multiple distinct pills, this is not a valid segmentation of one complete pill.

3. Black Treated as Background:
   Each pill is surrounded by black, confirming that they are interpreted as separate objects rather than fragments of a single one.

### Conclusion:

Because the image includes multiple complete objects instead of a single complete pill, the correct response is 'no'.
['no']

Now you need to refer to the previous examples to make a reasonable judgment on the following image:
Input Image: <image>
Object Description: {object_name} 