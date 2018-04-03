import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  title = 'app';
  linearModel: tf.Sequential;
  prediction: any;

  ngOnInit() {
     this.trainNewModel();
  }

  async trainNewModel() {
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs = tf.tensor1d([3.2, 4.4, 5.5, 6.71, 6.98, 7.168, 9.779, 6.182, 7.59, 2.16, 7.04, 6.11, 5.22,6.13]);
    const ys = tf.tensor1d([1.6, 2.7, 2.9, 3.19, 1.684,2.53, 3.336, 2.596, 2.53, 3.336, 2.596, 2.53,1.22, 2.82 ]);

    await this.linearModel.fit(xs, ys);
     console.log('modelo treinado!');

  }

  linearPrediction(val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1,1])) as any;
    this.prediction = Array.from(output.dataSync())[0];
  }

}
