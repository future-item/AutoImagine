In this task, you will receive a rendered image of a 3D scene and a description of the target object. 
There is a small red circle in the image. What you need to do is check if the red circle falls on the target object.
If the circle falls on the edge of the object and you find it difficult to judge, you should output ['no'].
Since the image is rendered by the edited 3D scene, please ignore artifacts like shadows, holes or floating spots.
In the image, all pure black area must be considered as illegal areas!
In the image, all pure black area must be considered as illegal areas!
In the image, all pure black area must be considered as illegal areas!
You should output the check result result in the last line in format: ['yes'] or ['no']. 

Example 1:
Input Image: A image contains several balls. A red circle falls on a ball.
Object Description: balls
Output: The target I'm looking for is balls. This circle lands on a ball, and it's not on the edge of the ball, and the ball is not pure black, so I should output "yes".
['yes']

Example 2:
Input Images: There are many dogs in the photo, sitting on the carpet, and a red circle falls at the edge of a dog and the carpet.
Object Description: dogs
Output:The target I'm looking for is dogs. The red circle falls at the edge between a dog and the carpet, although the dog and the carpet are not pure black
I find it difficult to judge, so I should output "no".
['no']

Example 3:
Input Images: There are several airplanes in the picture. Some of them are pure black, and a red circle is on one of the black airplanes.
Object Description: airplanes
Output:The red circle is on an airplane, but the airplane is pure black, which means this airplane is illegal, so I should output "no".
['no']

Now you need to refer to the previous examples to make a reasonable judgment on the following input: