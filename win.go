package main

import (
	"fmt"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
)

func main() {
	// A model exported with tf.saved_model.save()
	// automatically comes with the "serve" tag because the SavedModel
	// file format is designed for serving.
	// This tag contains the various functions exported. Among these, there is
	// always present the "serving_default" signature_def. This signature def
	// works exactly like the TF 1.x graph. Get the input tensor and the output tensor,
	// and use them as placeholder to feed and output to get, respectively.

	// To get info inside a SavedModel the best tool is saved_model_cli
	// that comes with the TensorFlow Python package.

	// e.g. saved_model_cli show --all --dir output/keras
	// gives, among the others, this info:

	// signature_def['serving_default']:
	// The given SavedModel SignatureDef contains the following input(s):
	//   inputs['inputs_input'] tensor_info:
	//       dtype: DT_FLOAT
	//       shape: (-1, 28, 28, 1)
	//       name: serving_default_inputs_input:0
	// The given SavedModel SignatureDef contains the following output(s):
	//   outputs['logits'] tensor_info:
	//       dtype: DT_FLOAT
	//       shape: (-1, 10)
	//       name: StatefulPartitionedCall:0
	// Method name is: tensorflow/serving/predict

	model := tg.LoadModel("./tfmodel", []string{"serve"}, nil)

	fakeInput, _ := tf.NewTensor([1][28][28][1]float32{})
	results := model.Exec([]tf.Output{
		model.Op("StatefulPartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("serving_default_inputs_input", 0): fakeInput,
	})

	predictions := results[0]
	fmt.Println(predictions.Value())
}
