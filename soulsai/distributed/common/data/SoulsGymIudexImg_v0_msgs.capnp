@0x80c4137c32fd5b9a;

struct DQNSample {
  obs @0 :List(List(List(UInt8)));
  action @1 :Int32;
  reward @2 :Float32;
  nextObs @3 :List(List(List(UInt8)));
  done @4 :Bool;
  info @5 :Info;
  epSteps @6 :Int32;
  modelId @7 :Text;
}

struct Info {
    allowedActions @0: List(Int32);
}

struct EpisodeInfo {
  epReward @0 :Float32;
  epSteps @1 :Int32;
  bossHp @2 :Float32;
  win @3 :Bool;
  eps @4 :Float32;
  modelId @5 :Text;
}

struct Telemetry {
  epReward @0 :Float32;
  epSteps @1 :Int32;
  totalSteps @2 :Int32;
  bossHp @3 :Float32;
  win @4 :Bool;
  eps @5 :Float32;
}