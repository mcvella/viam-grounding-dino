# grounding-dino modular service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a mcvella:vision:grounding-dino model.

This model leverages the [Grounding DINO computer vision model](https://github.com/IDEA-Research/GroundingDINO) to allow for image detection and querying.

The Grounding DINO model and inference will run locally, and therefore speed of inference is highly dependant on hardware.

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `rdk:vision:mcvella:vision:grounding-dino` model from the [`mcvella:vision:grounding-dino` module](https://app.viam.com/module/rdk/mcvella:vision:grounding-dino).

## Configure your vision

> [!NOTE]  
> Before configuring your vision, you must [create a machine](https://docs.viam.com/manage/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Components** subtab and click **Create component**.
Select the `vision` type, then select the `mcvella:vision:grounding-dino` model.
Enter a name for your vision and click **Create**.

On the new component panel, copy and paste the following attribute template into your vision’s **Attributes** box:

```json
{
  "model_id": "IDEA-Research/grounding-dino-tiny",
  "default_query": "house. car. zebra."
}
```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `rdk:vision:mcvella:vision:grounding-dino` visions:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `model_id` | string | Optional |  The HuggingFace model ID for the grounding DINO model |
| `default_query` | string | **Required** |  A list of classes to look for in images. Each class must end in a period. Note that multi-word classes will often be detected as the base word, for example "man cooking" might detect "man". |

### Example Configuration

```json
{
  "default_query": "road. car. stop sign."
}
```

## API

The grounding-dino resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

Note: if using this method, any cameras you are using must be set in the `depends_on` array for the service configuration, for example:

```json
      "depends_on": [
        "cam"
      ]
```

By default, the grounding-dino model will be use the "default_query".
If no "default_query" is specified no detections will occur.
If you want to look for different detection classes in an image, you can pass a different query as an extra parameter "query".
For example:

``` python
service.get_detections(image, extra={"query": "dog. cat. rat."})
```
