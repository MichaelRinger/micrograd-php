# micrograd-php
A PHP implementation of Andrej Karpathy's micrograd Autograd engine. 

## Usage
```php
$nn = new MLP(2, [3, 1]);
$xs = [
    [2.0, 3.0],
    [3.0, -1.0],
    [0.5, 1.0],
    [1.0, 1.0]
];
$ys = [1.0, -1.0, -1.0, 1.0];
$lr = 0.1;
$epochs = 10;
foreach (range(1, $epochs) as $e) {
    $loss = new Value(0);
    for ($i = 0; $i < count($xs); $i++) {
        $yPred = $nn->forward($xs[$i]);
        $loss = $loss->add($yPred->sub($ys[$i])->pow(2));
    }
    $loss->backward();

    foreach ($nn->parameters() as $p) {
        $p->data += -$p->grad * $lr;
    }
    $nn->zeroGrad();

    dump("Epoch: {$e}");
    dump("Loss: {$loss->data}");
    ($e != $epochs) ? dump("------") : '';
}
```
