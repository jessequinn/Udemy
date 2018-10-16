function greet(name, formatter) {
  return "Hello, " + formatter(name);
}

greet("Tim", function(name) {
  return name.toUpperCase();
});

greet("Tim", function(name) {
  return name + "!!!!!";
});

