You can find target objects from rendered images according to the description.

In this task, you will receive a rendered image of an object in the 3D scene, as well as the description of the target object. 
Your task is to check if the object shown in the image is the described object. If any small artifacts, render noise, or minor visual imperfections are present, ignore them unless they clearly resemble distinct, separate objects.
You should think step by step as follows.

1. The image should show a small single target object against a black background. Areas of the image that do not belong to the object are set to black. If not, you should judge it as ['no'] and end your check.
2. The objects in the image may be blurry or incomplete. Please ignore the artifacts and identify the object in the image from the overall color and shape.

For example, suppose the target object you want to judge is 'ball'. If the image shows only one ball-like object with no obvious defects against a black background, you should judge it as 'yes'. If the image shows a scene containing the ball (such as a table with a ball on it), or contains more than one ball, or shows a ball against a white background, you should judge it as 'no'.

You should think step by step. First analyze the object in the image, then give your reasons for your judgment, and finally output the check result in the last line in format: ['yes'] or ['no']. 