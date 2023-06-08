@0xb4dd05902eca5452;

struct Sample {
  obs @0 :List(Float32);
  action @1 :Int32;
  reward @2 :Float32;
  nextObs @3 :List(Float32);
  done @4 :Bool;
  info @5 :Info;
  modelId @6 :Text;
}


struct Info {
    allowedActions @0: List(Int32);
}

struct Telemetry {
  reward @0 :Float32;
  steps @1 :Int32;
  bossHp @2 :Float32;
  win @3 :Bool;
  eps @4 :Float32;
}
