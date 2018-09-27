// module patterns

// TODO: budget controller
var budgetController = (function () {

})();

// TODO: UI controller
var UIController = (function () {
    var DOMStrings = {
        inputType: '.add__type',
        inputDescription: '.add__description',
        inputValue: '.add__value',
        inputBtn: '.add__btn'
    };

    return {
        getInput: function () {
            return {
                type: document.querySelector(DOMStrings.inputType).value,
                description: document.querySelector(DOMStrings.inputDescription).value,
                value: document.querySelector(DOMStrings.inputValue).value
            }
        },
        getDOMstrings: function () {
            return DOMStrings;
        }
    }
})();

// TODO: global app controller, allow budgetController and UIController to interacte
var controller = (function (budgetCtrl, UICtrl) {
    var setupEventListeners = function () {
        var DOM = UICtrl.getDOMstrings();

        document.querySelector(DOM.inputBtn).addEventListener('click', ctrlAddItem);

        // keyCode deprecated
        document.addEventListener('keypress', function (event) {
            if (event.key === 13 || event.keyCode === 13 || event.which === 13) {
                ctrlAddItem();
            }
        });
    };

    var ctrlAddItem = function () {
        var input = UICtrl.getInput();

        console.log(input);
    };

    return {
        init: function() {
            console.log('Application has started.');
            setupEventListeners();
        }
    }
})(budgetController, UIController);

// TODO: start program
controller.init();